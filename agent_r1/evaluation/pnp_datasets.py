"""Frozen benchmark adapters; references are separate from actor inputs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re


@dataclass
class EvaluationTask:
    question_id: str
    actor_inputs: dict
    reference: dict


def dataset_files(path):
    path = Path(path).resolve()
    files = sorted(path.glob("test*.jsonl")) if path.is_dir() else [path]
    if not files or any(not item.is_file() for item in files):
        raise FileNotFoundError(f"Evaluation dataset is unavailable: {path}")
    return files


def dataset_hashes(path):
    result = {}
    for item in dataset_files(path):
        digest = hashlib.sha256()
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(item)] = digest.hexdigest()
    return result


def raw_rows(path):
    rows = []
    for item in dataset_files(path):
        if item.suffix == ".parquet":
            import pandas as pd
            rows.extend(pd.read_parquet(item).to_dict("records"))
        elif item.suffix == ".jsonl":
            with item.open(encoding="utf-8") as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
        elif item.suffix == ".json":
            content = json.loads(item.read_text(encoding="utf-8"))
            if not isinstance(content, list):
                raise ValueError("Evaluation JSON must contain a list of questions")
            rows.extend(content)
        else:
            raise ValueError("Supported dataset formats: parquet, jsonl, json")
    return rows


def _list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _image(value, root):
    from PIL import Image
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        value = value.get("path")
    if not value:
        raise ValueError("MATH-Vision requires image bytes or a resolvable image path")
    path = Path(value)
    return Image.open(path if path.is_absolute() else root / path).convert("RGB")


def load_tasks(benchmark, path, *, split, version, max_problems=None, load_images=True, task_slice=None):
    rows = raw_rows(path)
    if benchmark not in {"aime2025", "hotpotqa", "livecodebench", "mathvision"}:
        raise ValueError("Unknown evaluation benchmark")
    if benchmark == "aime2025" and (version != "2025" or len(rows) != 30):
        raise ValueError("AIME2025 requires version 2025 and all 30 questions before optional smoke-test selection")
    if benchmark == "livecodebench":
        if version != "release_v6" or len(rows) != 1055:
            raise ValueError("LiveCodeBench adapter requires frozen release_v6 (1055 questions)")
    # Check the complete snapshot before selecting a subset or worker shard.
    # Duplicates outside a smoke test must not turn into an apparently valid run.
    def question_id(index, row):
        extra = row.get("extra_info") or {}
        if benchmark == "aime2025":
            return str(row.get("problem_idx", row.get("id", extra.get("question_id", index))))
        if benchmark == "hotpotqa":
            return str(extra["question_id"])
        if benchmark == "livecodebench":
            return str(row["question_id"])
        return str(row.get("id", extra.get("question_id", index)))
    all_ids = [question_id(index, row) for index, row in enumerate(rows)]
    if not all_ids or len(all_ids) != len(set(all_ids)):
        raise ValueError("Benchmark must have nonempty, unique question IDs")
    tasks = []
    root = Path(path).resolve().parent
    selection = list(enumerate(rows))
    if max_problems is not None:
        if max_problems < 1:
            raise ValueError("max_problems must be positive")
        selection = selection[:max_problems]
    if task_slice is not None:
        selection = selection[task_slice]
    for index, row in selection:
        extra = row.get("extra_info") or {}
        if benchmark == "aime2025":
            from agent_r1.evaluation.answers import aime_integer
            question = row.get("problem")
            if question is None:
                prompt = row.get("prompt")
                if prompt is None:
                    raise ValueError("AIME2025 question is missing")
                messages = _list(prompt)
                user_contents = [message.get("content") for message in messages if message.get("role") == "user"]
                if len(user_contents) != 1:
                    raise ValueError("Prepared AIME2025 rows require exactly one user question")
                question = user_contents[0]
            if not isinstance(question, str) or not question.strip():
                raise ValueError("AIME2025 question is missing")
            reference = row.get("answer")
            if reference is None:
                reference = (row.get("reward_model") or {}).get("ground_truth")
            if reference is None or aime_integer(str(reference)) is None:
                raise ValueError("Invalid AIME reference integer")
            task = EvaluationTask(all_ids[index], {"problem": question}, {"answer": str(reference)})
        elif benchmark == "hotpotqa":
            if "prompt" not in row:
                raise ValueError("HotpotQA requires prepared validation rows aligned with the evidence sidecar")
            question = str(_list(row["prompt"])[0]["content"])
            qid = str(extra["question_id"])
            if extra.get("split", split) != split:
                raise ValueError("HotpotQA row split does not match the selected evaluation split")
            reference = (row.get("reward_model") or {}).get("ground_truth")
            answers = _list(reference) if isinstance(reference, (list, tuple)) or hasattr(reference, "tolist") else [reference]
            if reference is None:
                raise ValueError("Missing HotpotQA answer")
            task = EvaluationTask(qid, {"raw_prompt": [{"role": "user", "content": question}],
                                      "extra_info": {"question_id": qid, "split": str(extra.get("split", split)),
                                                     "index": int(extra.get("index", index))}},
                                  {"answers": [str(answer) for answer in answers]})
        elif benchmark == "livecodebench":
            from agent_r1.evaluation.pnp_code import problem_tests
            public, private, function_name = problem_tests(row)
            qid = str(row["question_id"])
            question = str(row["question_content"])
            starter = str(row.get("starter_code") or "")
            if starter:
                fence = chr(96) * 3
                question += "\nComplete the following starter code:\n" + fence + "python\n" + starter + "\n" + fence
            else:
                question += "\nRead input from stdin and write output to stdout."
            task = EvaluationTask(qid, {"raw_prompt": [{"role": "user", "content": question}],
                                      "extra_info": {"question_id": qid},
                                      "public_tests": public, "function_name": function_name},
                                  {"public_tests": public, "private_tests": private, "function_name": function_name})
        else:
            question = str(row.get("question", row.get("problem", ""))).strip()
            if not question:
                raise ValueError("MATH-Vision question is missing")
            options = _list(row["options"]) if row.get("options") is not None else []
            question = re.sub(r"<image\d*>", "", question, flags=re.I).strip()
            if options:
                question += "\nOptions: " + " ".join(f"{chr(65+i)}. {value}" for i, value in enumerate(options))
            payload = row.get("decoded_image", row.get("image"))
            if payload is None and row.get("images") is not None:
                images = _list(row["images"])
                if len(images) != 1:
                    raise ValueError("MATH-Vision PNP requires exactly one root image")
                payload = images[0]
            image = _image(payload, root) if load_images else None
            qid = str(row.get("id", extra.get("question_id", index)))
            task = EvaluationTask(qid, {"raw_prompt": [{"role": "user", "content": [
                                       {"type": "image", "image": image, "max_pixels": 1048576},
                                       {"type": "text", "text": question}]}],
                                       "extra_info": {"question_id": qid}},
                                  {"answer": str(row["answer"]), "options": [str(value) for value in options]})
        tasks.append(task)
    if (not tasks and task_slice is None) or len({task.question_id for task in tasks}) != len(tasks):
        raise ValueError("Benchmark must have nonempty, unique question IDs")
    return tasks, len(rows)
