"""Independent, frozen-protocol PNP evaluation on four held-out benchmarks."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from agent_r1.evaluation.pnp_datasets import dataset_hashes, load_tasks
from agent_r1.evaluation.pnp_protocol import PROFILES, evaluation_profile, mathvision_score, qa_scores, summarize_records
from agent_r1.evaluation.official import grader_identity

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_groups(devices, tensor_parallel_size):
    values = devices.split(",")
    if not values or any(not value.isdigit() for value in values) or len(set(values)) != len(values):
        raise ValueError("--gpus requires unique numeric device IDs")
    if tensor_parallel_size < 1 or len(values) % tensor_parallel_size:
        raise ValueError("GPU count must be divisible by tensor parallel size")
    return [values[index:index + tensor_parallel_size] for index in range(0, len(values), tensor_parallel_size)]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("benchmark", nargs="?", choices=tuple(PROFILES))
    result.add_argument("--model", "--model-path", dest="model", type=Path)
    result.add_argument("--checkpoint", type=Path, help="Source checkpoint identity; --model must be a vLLM-loadable export")
    result.add_argument("--dataset", type=Path)
    result.add_argument("--dataset-version", help="Frozen year/release/config, e.g. 2025 or release_v6")
    result.add_argument("--dataset-revision", help="Pinned source commit or local snapshot identifier (file hashes are also recorded)")
    result.add_argument("--split", choices=("test", "validation"), help="HotpotQA: validation; other benchmarks: test")
    result.add_argument("--output", type=Path, help="New run directory; defaults to logs/pnp_eval_<benchmark>_<timestamp>")
    result.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    result.add_argument("--tensor-parallel-size", type=int, default=1)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    result.add_argument("--total-output-tokens", type=int)
    result.add_argument("--per-step-tokens", type=int)
    result.add_argument("--prompt-limit", type=int)
    result.add_argument("--context-limit", type=int)
    result.add_argument("--max-problems", type=int, help="Smoke-test subset, explicitly marked as such")
    result.add_argument("--corpus-dir", type=Path, default=ROOT / "data/corpus/hotpotqa_corpus")
    result.add_argument("--evidence-sidecar", type=Path)
    result.add_argument("--embedding-model", default=os.environ.get("HOTPOTQA_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))
    result.add_argument("--timeout-seconds", type=float, default=6.0)
    result.add_argument("--preflight-only", action="store_true", help="Check data/resources/dependencies without allocating GPUs")
    result.add_argument("--write-tables", type=Path, help="Write four LaTeX tables from shared defaults; does not run evaluation")
    result.add_argument("--worker-rank", type=int, help=argparse.SUPPRESS)
    return result


def profile_from_args(args):
    return evaluation_profile(args.benchmark, total_output_tokens=args.total_output_tokens,
                              per_step_tokens=args.per_step_tokens, prompt_limit=args.prompt_limit,
                              context_limit=args.context_limit)


def resources_from_args(args):
    return {"corpus_dir": str(args.corpus_dir.resolve()),
            "evidence_sidecar": str((args.evidence_sidecar or args.corpus_dir / "hotpotqa_evidence_v1.sqlite3").resolve()),
            "embedding_model": args.embedding_model, "timeout_seconds": args.timeout_seconds}


def preflight(args):
    missing = [name for name in ("vllm", "transformers", "torch") if importlib.util.find_spec(name) is None]
    if args.benchmark != "aime2025":
        missing += [name for name in ("verl", "ray", "omegaconf") if importlib.util.find_spec(name) is None]
    if args.benchmark == "hotpotqa":
        missing += [name for name in ("faiss", "FlagEmbedding") if importlib.util.find_spec(name) is None]
        resources = resources_from_args(args)
        for path in (args.corpus_dir / "index.bin", args.corpus_dir / "hpqa_corpus.jsonl",
                     Path(resources["evidence_sidecar"])):
            if not path.is_file():
                missing.append(str(path))
    if args.benchmark == "livecodebench" and not shutil.which("bwrap"):
        missing.append("bubblewrap (bwrap)")
    if args.model is None or not (args.model / "config.json").is_file():
        missing.append("vLLM-loadable model directory with config.json")
    elif not list(args.model.glob("*.safetensors")):
        missing.append("exported model safetensors (FSDP shards must be merged before evaluation)")
    if missing:
        raise RuntimeError("Evaluation prerequisites missing: " + ", ".join(missing))
    grader_identity(args.benchmark)
    if args.benchmark == "livecodebench":
        from agent_r1.evaluation.pnp_code import grade_submission
        # An installed executable can still be unusable in containers. Detect
        # that before spending GPU time or counting infrastructure failures as
        # incorrect model answers.
        result = grade_submission("print(input())", {
            "public_tests": [{"input": "public\n", "output": "public\n"}],
            "private_tests": [{"input": "private\n", "output": "private\n"}],
            "function_name": None}, args.timeout_seconds)
        if not result["is_correct"]:
            raise RuntimeError("LiveCodeBench sandbox preflight failed: " + result["grading_reason"])


def grade_record(benchmark, record, reference, timeout_seconds):
    if benchmark == "hotpotqa":
        return qa_scores(record["submitted_answer"] if record["terminal_eligible"] else None, reference["answers"])
    if benchmark == "mathvision":
        return {"is_correct": bool(record["terminal_eligible"] and mathvision_score(
            record["submitted_answer"], reference["answer"], reference["options"]))}
    from agent_r1.evaluation.pnp_code import grade_submission
    return grade_submission(record["submitted_answer"] if record["terminal_eligible"] else None,
                            reference, timeout_seconds)


def math_records(engine, tokenizer, sampling_factory, tasks, profile):
    from agent_r1.evaluation.pnp_math import generate_trajectories, selected_aime_integer
    from agent_r1.evaluation.answers import aime_integer, final_answer_record
    from recipes.deepscaler.trajectory_reward import verify_process
    states = generate_trajectories(
        engine, tokenizer, sampling_factory, [task.actor_inputs["problem"] for task in tasks],
        max_steps=profile.max_steps, total_tokens=profile.total_output_tokens,
        initial_limit=profile.prompt_limit, context_limit=profile.context_limit,
        per_step_tokens=profile.per_step_tokens)
    records = []
    for task, state in zip(tasks, states, strict=True):
        integer = selected_aime_integer(state["selected_answer"])
        segments = []
        for step in state["steps"]:
            parsed = final_answer_record(step["response"])
            reasoning = "" if step["mode"] == "submission" else parsed["reasoning"] if parsed else step["response"]
            segments.append((step["turn"], reasoning))
        verification = verify_process(segments, question=task.actor_inputs["problem"])
        records.append({
            "question_id": task.question_id, "submitted_answer": state["selected_answer"],
            "predicted_integer": integer, "terminal_eligible": integer is not None,
            "is_correct": integer is not None and integer == aime_integer(task.reference["answer"]),
            "termination_reason": state["termination_reason"], "policy_step_count": len(state["steps"]),
            "generated_output_tokens": state["response_tokens"], "steps": state["steps"],
            "verification": {"verified_process_reward": verification.process_reward,
                             "audit_completed": True, "audit": verification.audit}})
    return records


def worker(args, tasks, profile):
    from transformers import AutoTokenizer, AutoProcessor
    from vllm import LLM, SamplingParams
    from agent_r1.evaluation.pnp_runtime import BudgetedServer, generate_task
    groups = gpu_groups(args.gpus, args.tensor_parallel_size)
    selected = tasks
    if not selected:
        (args.output / f"replica_{args.worker_rank}.jsonl").write_text("")
        return
    os.environ["HOTPOTQA_REWARD_ARM"] = "A9_CERT_MIX"
    os.environ["HOTPOTQA_EMBEDDING_DEVICE"] = "cpu"
    os.environ["HOTPOTQA_FORMAL_EXPERIMENT"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True) if args.benchmark == "mathvision" else None
    engine_args = dict(model=str(args.model), tensor_parallel_size=args.tensor_parallel_size,
                       dtype=profile.precision, max_model_len=profile.context_limit,
                       gpu_memory_utilization=args.gpu_memory_utilization,
                       enforce_eager=True, generation_config="vllm", seed=profile.seed,
                       max_num_seqs=16, max_num_batched_tokens=profile.context_limit,
                       trust_remote_code=True)
    if args.benchmark == "mathvision":
        engine_args["limit_mm_per_prompt"] = {"image": profile.max_steps}
    else:
        engine_args.update(language_model_only=True, mm_processor_cache_gb=0)
    engine = LLM(**engine_args)
    resources = resources_from_args(args)
    with (args.output / f"replica_{args.worker_rank}.jsonl").open("w", encoding="utf-8") as stream:
        if args.benchmark == "aime2025":
            records = math_records(engine, tokenizer, SamplingParams, selected, profile)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
        else:
            for task in selected:
                server = BudgetedServer(engine, SamplingParams, profile)
                record = asyncio.run(generate_task(
                    args.benchmark, task, tokenizer, processor, server, profile, resources))
                record.update(grade_record(args.benchmark, record, task.reference, args.timeout_seconds))
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"question={task.question_id} correct={record['is_correct']} "
                      f"stop={record['termination_reason']} tokens={record['generated_output_tokens']}", flush=True)


def write_tables(path):
    lines = ["# PNP evaluation configuration", "",
             "These are shared evaluator defaults, not measurements from completed runs. "
             "All four benchmarks reuse PNP policy/tool interfaces. Run manifests record any explicit overrides.",
             "", "Accuracy uses the final submitted answer/program, independent of process reward. "
             "Missing or invalid submissions remain in the denominator. References and hidden tests are grading-only.",
             "", "Total completion budgets count generated tokens across all interaction steps; "
             "context limits also account for observations and image tokens. "
             "The default node has eight NVIDIA A40 GPUs. Tensor parallel size defaults to 1 per replica; "
             "the selected GPU count determines replica count.",
             "", chr(96) * 3 + "latex", r"% Requires \usepackage{booktabs}", ""]
    names = {"aime2025": "AIME2025", "hotpotqa": "HotpotQA",
             "livecodebench": "LiveCodeBench", "mathvision": "MATH-Vision"}
    metrics = {"aime2025": "Integer accuracy", "hotpotqa": "Answer EM (F1 auxiliary)",
               "livecodebench": "Pass@1", "mathvision": "Answer accuracy"}
    for benchmark, profile in PROFILES.items():
        lines += [r"\begin{table}[!ht]", r"\centering", r"\begin{tabular}{@{}ll@{}}",
                  r"\toprule", r"\textbf{Parameter} & \textbf{Value} \\", r"\midrule"]
        rows = [("Sampling temperature", "0.0"), ("Top-$p$ / Top-$k$", "1.0 / -1"),
                ("Trajectories per problem $n$", "1"), ("Evaluation metric", metrics[benchmark]),
                ("Total completion budget", f"{profile.total_output_tokens} tokens"),
                ("Per-step generation cap", f"{profile.per_step_tokens} tokens"),
                ("Maximum interaction steps", str(profile.max_steps)),
                ("Initial prompt limit" if benchmark == "aime2025" else "Per-step prompt limit",
                 f"{profile.prompt_limit} tokens"), ("Context limit", f"{profile.context_limit} tokens"),
                ("Inference engine", "vLLM"), ("Mode", "Offline Python API"),
                ("Tensor parallel size", "1 per replica"), ("Precision / Seed", "bfloat16 / 42")]
        lines += [f"{key} & {value} " + r"\\" for key, value in rows]
        lines += [r"\bottomrule", r"\end{tabular}",
                  r"\caption{PNP evaluation configuration for " + names[benchmark] + ".}",
                  r"\label{tab:pnp_eval_" + benchmark + "}", r"\end{table}", r"\newpage", ""]
    lines += [chr(96) * 3, "", "Frozen dataset identity (version, revision, split, question IDs and file SHA256), "
              "checkpoint identity, code hashes, budgets and deployment settings are recorded per run.",
              "", "HotpotQA uses the prepared validation split aligned with the retrieval evidence sidecar. "
              "LiveCodeBench uses release_v6; public tests may be used during interaction, and all public/private "
              "tests grade the final submission after generation using the original pinned LiveCodeBench checker. "
              "MATH-Vision uses the original pinned MATH-V script for answer extraction, choice handling, "
              "latex2sympy equivalence, and numeric comparison. Upstream scoring sources remain unchanged.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    if args.write_tables:
        write_tables(args.write_tables)
        print(f"Saved {args.write_tables}")
        return
    for name in ("benchmark", "model", "dataset", "dataset_version", "dataset_revision"):
        if getattr(args, name) is None:
            command_parser.error(f"{name.replace('_', '-')} is required for evaluation")
    if args.dataset_revision.lower() in {"main", "master", "latest", "head"}:
        command_parser.error("--dataset-revision must identify a frozen snapshot")
    if not 0 < args.gpu_memory_utilization < 1 or args.timeout_seconds <= 0:
        command_parser.error("GPU memory utilization must be in (0,1), and timeout must be positive")
    expected_split = "validation" if args.benchmark == "hotpotqa" else "test"
    args.split = args.split or expected_split
    if args.split != expected_split:
        command_parser.error(f"{args.benchmark} requires the {expected_split} split")
    args.model, args.dataset = args.model.resolve(), args.dataset.resolve()
    groups = gpu_groups(args.gpus, args.tensor_parallel_size)
    profile = profile_from_args(args)
    if args.worker_rank is not None:
        if not 0 <= args.worker_rank < len(groups):
            command_parser.error("Invalid evaluation worker rank")
        recorded = json.loads((args.output / "run_manifest.json").read_text())
        if dataset_hashes(args.dataset) != recorded["dataset"]["file_sha256"]:
            raise RuntimeError("Dataset changed after evaluation manifest creation")
    tasks, full_count = load_tasks(
        args.benchmark, args.dataset, split=args.split, version=args.dataset_version,
        max_problems=args.max_problems, load_images=args.worker_rank is not None,
        task_slice=slice(args.worker_rank, None, len(groups)) if args.worker_rank is not None else None)
    if args.worker_rank is not None:
        worker(args, tasks, profile)
        return
    preflight(args)
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "selected_questions": len(tasks),
                          "dataset_questions": full_count, "profile": profile.record()}, indent=2))
        return
    args.output = args.output or ROOT / "logs" / (
        "pnp_eval_" + args.benchmark + "_" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a new output directory; evaluation results are never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = ["agent_r1/evaluation/" + name for name in
                    ("pnp.py", "pnp_protocol.py", "pnp_datasets.py", "pnp_runtime.py", "pnp_code.py", "pnp_math.py",
                     "official.py", "official_worker.py")]
    source_paths.append("third_party/official_graders/manifest.json")
    official_grader = grader_identity(args.benchmark)
    source_paths.extend("third_party/official_graders/" + args.benchmark + "/" + relative
                        for relative in official_grader.get("files_sha256", {}))
    source_paths += ["scripts/deepscaler/pnp_aime.py", "recipes/deepscaler/prompts.py",
                     "recipes/deepscaler/submission.py", "recipes/hotpotqa_a9/agent_flow.py",
                     "recipes/taco_a9/agent_flow.py", "recipes/vision_r1/agent_flow.py"]
    for directory in ("agent_r1/verifier", "recipes/deepscaler", "recipes/hotpotqa_a9",
                      "recipes/taco_a9", "recipes/vision_r1", "recipes/livecodebench_a0"):
        source_paths.extend(str(path.relative_to(ROOT)) for path in (ROOT / directory).glob("*.py"))
    source_paths += ["agent_r1/agent_flow/agent_flow.py", "agent_r1/env/tool_format.py",
                     "recipes/hotpotqa/evidence.py", "recipes/hotpotqa/env/search_tool.py",
                     "recipes/hotpotqa_lr/protocol.py", "recipes/hotpotqa_lr/verifier.py"]
    package_versions = {}
    for name in ("torch", "vllm", "transformers", "verl", "ray", "Pillow"):
        try:
            package_versions[name] = version(name)
        except PackageNotFoundError:
            package_versions[name] = "unknown"
    manifest = {"configuration": profile.record(), "model_path": str(args.model),
                "source_checkpoint": str((args.checkpoint or args.model).resolve()),
                "dataset": {"path": str(args.dataset), "version": args.dataset_version,
                            "revision": args.dataset_revision, "split": args.split,
                            "file_sha256": dataset_hashes(args.dataset),
                            "selected_question_ids": [task.question_id for task in tasks],
                            "full_question_count": full_count, "smoke_test": len(tasks) < full_count},
                "deployment": {"gpu_groups": groups, "tensor_parallel_size_per_replica": args.tensor_parallel_size,
                               "gpu_memory_utilization": args.gpu_memory_utilization},
                "resources": resources_from_args(args),
                "package_versions": package_versions,
                "official_grader": official_grader,
                "model_file_sha256": {str(path.relative_to(args.model)): next(iter(dataset_hashes(path).values()))
                                      for path in sorted(args.model.glob("*"))
                                      if path.is_file() and path.suffix in {".json", ".safetensors", ".py"}},
                "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in sorted(set(source_paths))}}
    if args.benchmark == "hotpotqa":
        resource_paths = (args.corpus_dir / "index.bin", args.corpus_dir / "hpqa_corpus.jsonl",
                          Path(manifest["resources"]["evidence_sidecar"]))
        manifest["resources"]["file_sha256"] = {
            str(path.resolve()): next(iter(dataset_hashes(path).values())) for path in resource_paths}
    write_json(args.output / "run_manifest.json", manifest)
    processes = []
    started = time.monotonic()
    forwarded = [args.benchmark, "--model", str(args.model), "--dataset", str(args.dataset),
                 "--dataset-version", args.dataset_version, "--dataset-revision", args.dataset_revision,
                 "--split", args.split, "--output", str(args.output), "--gpus", args.gpus,
                 "--tensor-parallel-size", str(args.tensor_parallel_size),
                 "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                 "--corpus-dir", str(args.corpus_dir), "--embedding-model", args.embedding_model,
                 "--timeout-seconds", str(args.timeout_seconds)]
    for name in ("total_output_tokens", "per_step_tokens", "prompt_limit", "context_limit",
                 "max_problems", "evidence_sidecar"):
        if getattr(args, name) is not None:
            forwarded += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    try:
        for rank, devices in enumerate(groups):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devices),
                   "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_LOGGING_LEVEL": "WARN",
                   "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"}
            with (args.output / f"replica_{rank}.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-m", "agent_r1.evaluation.pnp",
                    *forwarded, "--worker-rank", str(rank)], env=env, cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            print(f"Started evaluation replica {rank} on GPUs {','.join(devices)}: pid={process.pid}", flush=True)
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("Evaluation worker failed; inspect replica logs. No accuracy was published.")
            time.sleep(2)
        if any(process.returncode for process in processes):
            raise RuntimeError("Evaluation worker failed; no accuracy was published.")
        records = []
        for rank in range(len(groups)):
            records += [json.loads(line) for line in (args.output / f"replica_{rank}.jsonl").read_text().splitlines()]
        order = {task.question_id: index for index, task in enumerate(tasks)}
        records.sort(key=lambda row: order[row["question_id"]])
        summary = summarize_records(records, list(order), profile)
        summary.update(elapsed_seconds=time.monotonic() - started, smoke_test=len(tasks) < full_count)
        (args.output / "results.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
