# Shared PNP benchmark evaluation

Use `bash run_pnp_eval.sh BENCHMARK ...` from the repository root. The wrapper
selects the `pnp` environment; `PYTHON_BIN` can select another explicit interpreter.
This is a standalone evaluator using vLLM's offline Python API. It does not start
a trainer, launch Ray, or modify training runs. The generated
[four configuration tables](pnp_evaluation_config.md) read the same defaults as the evaluator.

## Main protocol

Every selected question receives one complete agent trajectory with greedy
decoding (`temperature=0`, `top_p=1`, `top_k=-1`, `n=1`), bfloat16, and seed 42.
The agent retains its task's PNP prompts, tools, immutable artifacts, and submission
protocol. Code and vision evaluation use the corresponding PNP flows rather than
their historical A0 evaluators. Mathematics uses the training flow's bounded
continuation and final-submission helpers.

Accuracy is the mean binary correctness of final submissions across **all selected
questions**. Missing/invalid submissions and budget exhaustion without submission
score zero. Process verification is recorded separately and survives missing final
submissions; it cannot increase accuracy. Dataset, backend, or sandbox infrastructure
errors abort the evaluation instead of publishing an incomplete accuracy.

| Benchmark | Final metric | Steps | Total generated tokens | Per step | Prompt limit | Context limit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AIME2025 | Integer accuracy | 5 | 4096 | 820 | 2048 initial | 8192 |
| HotpotQA validation | Answer EM; F1 auxiliary | 4 | 4096 | 1024 | 8192 each step | 12288 |
| LiveCodeBench release_v6 | Pass@1 | 5 | 10240 | 2048 | 8192 each step | 12288 |
| MATH-Vision | Answer accuracy | 3 | 6144 | 2048 | 8192 each step | 12288 |

Total output budgets count every token generated across the trajectory, including
reasoning and tool-call syntax. Observations and visual tokens consume context,
not generated-output budget. In mathematics the final generation may use several
bounded chunks within one interaction step. Explicit budget overrides are recorded.
Match dataset snapshots, tools, graders, and budgets for every policy compared
within a task, including base models and trained checkpoints. The shared defaults
do not silently enlarge the 9B budget.

## Model and data prerequisites

`--model` must name a local Hugging Face model export with configuration, tokenizer,
and safetensors. Merge FSDP shards and, when applicable, merge the LoRA adapter into
the base weights before evaluation. `--checkpoint` records the source training
checkpoint identity; it does not convert or load trainer shards.

Supply a frozen `--dataset-version`, `--dataset-revision`, and split. Floating
revisions such as `main` or `latest` are rejected. A pinned upstream commit or a
named local snapshot is accepted; actual file SHA256 values are also recorded.
JSONL, JSON arrays, and parquet are supported. LiveCodeBench directories contain
`test*.jsonl`; other examples use explicit files.

Examples below are commands for future evaluations. Replace paths and the
snapshot identifiers with the actual sources; these are not claims of completed runs.

```bash
# Add --preflight-only to any command to validate prerequisites without GPUs.
bash run_pnp_eval.sh aime2025 \
  --model models/Qwen3.5-4B \
  --dataset data/corpus/aime2025/test.parquet \
  --dataset-version 2025 --dataset-revision LOCAL_AIME2025_SNAPSHOT \
  --gpus 0,1,2,3

bash run_pnp_eval.sh hotpotqa \
  --model /path/to/exported-hotpotqa-model \
  --dataset /path/to/prepared/validation.parquet \
  --dataset-version distractor --dataset-revision PINNED_HOTPOTQA_SNAPSHOT \
  --corpus-dir /path/to/hotpotqa_corpus \
  --evidence-sidecar /path/to/hotpotqa_evidence_v1.sqlite3 \
  --embedding-model /path/to/frozen-bge-model --gpus 0,1,2,3

bash run_pnp_eval.sh livecodebench \
  --model /path/to/exported-code-model --dataset /path/to/release_v6 \
  --dataset-version release_v6 --dataset-revision PINNED_LCB_COMMIT \
  --gpus 0,1,2,3

bash run_pnp_eval.sh mathvision \
  --model /path/to/exported-vision-model --dataset /path/to/mathvision/test.parquet \
  --dataset-version full --dataset-revision PINNED_MATHVISION_COMMIT \
  --gpus 0,1,2,3
```

AIME2025 requires `--dataset-version 2025` and the full 30-question I + II snapshot
before optional subset selection. It accepts raw `problem` / `answer` rows or
prepared `prompt` / `reward_model.ground_truth` rows, with references kept out of
policy inputs. Supply an AIME2025 snapshot and record its pinned revision.

LiveCodeBench requires the full 1055-question release_v6 snapshot. HotpotQA uses
prepared validation rows aligned with the retrieval evidence sidecar and requires
FAISS and FlagEmbedding. Record any prepared validation subset in its snapshot
identity. Standard answers and gold supporting facts are excluded from policy
inputs. Retrieval remains the existing corpus-search interface.

LiveCodeBench requires a working `bwrap` sandbox; preflight checks a known correct
program before GPU allocation. Public tests are exposed by the PNP testing tool.
Hidden tests are supplied only to the final grader after generation. Both stdio
and functional problems are supported. Final pass@1 requires all public and
private tests to pass. The repository's frozen local grader is used: its stdio
comparison strips surrounding whitespace per line, and its functional comparison
uses the existing runner. This is not a claim of exact parity with every upstream
LiveCodeBench checker, including numerical tolerance rules.

MATH-Vision preserves the root image and answer options, and exposes inspection
and explicit submission tools. Answer checking against the reference is unavailable
during interaction, including in the legacy ReAct entrypoint. Choice letters or
corresponding option values are accepted. String and bounded scalar/tuple numeric
equivalence follow the two-decimal MATH-V convention. Unsupported symbolic
equivalences fail closed; this local grader is not bit-for-bit identical to the
upstream latex2sympy checker.

## Deployment and artifacts

The default `--gpus 0,1,2,3 --tensor-parallel-size 1` starts four independent replicas,
each evaluating its assigned questions exactly once. `--tensor-parallel-size 2`
instead starts two replicas, each using two GPUs. The memory fraction defaults to
0.3. Choose available GPUs explicitly when running alongside other experiments.

Each new `logs/pnp_eval_<benchmark>_<timestamp>/` directory contains a frozen
`run_manifest.json`, per-replica logs/results, consolidated `results.jsonl`, and
`summary.json`. The manifest records dataset identity, selected question IDs,
model/code hashes, package versions, deployment, and budgets. For HotpotQA it
also hashes the retrieval corpus, index, and evidence sidecar. Results include
final eligibility, stop reason, output-token count, generated actions, and
separate process-verification diagnostics. `--max-problems N` marks the result as
a smoke test; a partially failed worker run never produces a summary accuracy.
Existing nonempty result directories are not overwritten.

Regenerate the four LaTeX tables after changing protocol defaults:

```bash
bash run_pnp_eval.sh --write-tables docs/pnp_evaluation_config.md
```

## Reference implementations

The task interfaces follow the repository's PNP AgentFlows, built on
[Agent-R1](https://github.com/AgentR1/Agent-R1). Answer normalization and F1 follow
the [HotpotQA evaluation script](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py).
The benchmark references are the official
[LiveCodeBench repository](https://github.com/LiveCodeBench/LiveCodeBench) and
[MATH-V repository](https://github.com/mathllm/MATH-V). Their task metrics remain
distinct; consistent evaluation here means matched policy interfaces and a fixed
single-trajectory protocol, with the local grader differences disclosed above.
