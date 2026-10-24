# Run entrypoints

Run these commands from the repository root, with the veRL environment, model
files, prepared datasets, and each task's services available. See the
[paper contract](../docs/paper-contract.md) for the matched profiles and
[shared settings](common/README.md) for 4B/9B model adaptation.

## Paper training

PNP has a common entrypoint at the repository root:

```bash
bash run_pnp.sh deepscaler --model Qwen3.5-4B --gpus 0,1,2,3,4,5,6,7
bash run_pnp.sh hotpotqa --model Qwen3.5-9B --gpus 0,1,2,3,4,5,6,7
```

Dataset launchers and agent configurations are named `pnp_<dataset>.sh` and
`pnp_<dataset>.yaml`. Model profiles are `models/Qwen3.5-4B.sh` and
`models/Qwen3.5-9B.sh`; shared settings are in `common/pnp.sh`.
The formal training node uses eight NVIDIA A40 GPUs for both model scales.
The 4B mathematics profile defaults to all eight GPUs, vLLM memory fraction 0.30,
32 concurrent sequences, checkpointing every 10 updates, and no periodic evaluation.
Frozen-judge comparisons allocate seven actor GPUs and one judge GPU on this node.

| Domain | Outcome-only GRPO | Verifier GRPO |
| --- | --- | --- |
| DeepScaleR | `bash scripts/deepscaler/run_deepscaler_a1.sh` | `bash scripts/deepscaler/pnp_deepscaler.sh` |
| HotpotQA | `bash scripts/hotpotqa/run_a1.sh` | `bash scripts/hotpotqa/pnp_hotpotqa.sh` |
| TACO | `bash scripts/taco/run_a1_terminal.sh` | `bash scripts/taco/pnp_taco.sh` |
| Vision-R1 | `bash scripts/vision_r1/run_a1_terminal.sh` | `bash scripts/vision_r1/pnp_vision_r1.sh` |

```bash
# Frozen process judge; domain: deepscaler, hotpotqa, taco, or vision.
bash scripts/llm_judge/run_grpo_4b_judge_9b.sh deepscaler
# Terminal-only PPO through the same domain interface.
bash scripts/ppo/run_terminal.sh deepscaler
# HotpotQA privileged gold-evidence ablations.
bash scripts/hotpotqa/run_a2.sh
bash scripts/hotpotqa/run_a3.sh
```

Math defaults to at most five reasoning turns, shared by A1, A9, Judge, PPO
and AIME evaluation. Use `DEEPSCALER_MAX_STEPS=8` for more turns or `=1` for
an optional single-generation setting. The total generated-token budget
remains 4096 (4B) or 5120 (9B); no correctness feedback enters this flow.
The reported mathematics results use the default five-generation protocol.

The shared `models/Qwen3.5-9B.sh DATASET` profile includes the previous HotpotQA
and TACO 9B memory settings. Domain launchers share model adaptation from
`common/pnp.sh`. `run_a7.sh` is the
retained weak-execution control; `pnp_hotpotqa_8gpu.sh` is an alternative hardware wrapper
for the same HotpotQA verifier contract.

## Evaluation

Use `bash run_pnp_eval.sh BENCHMARK ...` for matched single-trajectory PNP
evaluation of `aime2026`, `hotpotqa`, `livecodebench`, and `mathvision`.
The [shared guide](../docs/pnp_evaluation.md) documents the required frozen
dataset revisions, model exports, budgets, final graders, and CPU preflight.
The entries below include compatibility wrappers and historical baselines.

| Task | Entry point |
| --- | --- |
| AIME2026 PNP compatibility wrapper | `scripts/deepscaler/pnp_aime.py`, delegating to the shared evaluator |
| AIME 2026 | `scripts/deepscaler/eval_aime2026.py`; paired A0 wrapper `scripts/deepscaler/run_aime2026_a0.sh` |
| HotpotQA A0 | `scripts/hotpotqa/run_a0.sh` |
| HotpotQA PNP compatibility wrapper | `scripts/hotpotqa/pnp_hotpotqa_eval.sh`, delegating to the shared evaluator |
| MATH-Vision A0 | `scripts/mathvision/run_a0_react.sh`; standalone evaluator `recipes/mathvision/evaluate_react.py` |
| LiveCodeBench A0 | `recipes/livecodebench_a0/evaluate_a0.py` |
| Audit metrics from JSONL | `python -m agent_r1.evaluation.summarize validation.jsonl --output metrics.json` |

These are the existing evaluation utilities, not a claim that all paper scores
have been reproduced. Task validation and held-out benchmark evaluation remain
separate protocols.

## Extensions and environment patches

- `extensions/deepscaler/`: multi-turn answer-checking ToolEnv experiments. The incomplete DeepMath-103K launcher was removed because its recipe dependency was absent.
- `extensions/hotpotqa_lr/`: A8 local-reasoning training, calibration, validation, and diagnostics.
- `extensions/hotpotqa/`: old GPU/resume/storage-specific profiles. `run_a9_fixed.sh` still names a historical resume checkpoint; it is not a fresh-run default.
- `extensions/taco/`: reduced-data smoke tests and separated actor/rollout profiles.
- `patches/`: explicit environment patch scripts. Inspect their expected installed source/version before applying; moving them does not apply any patch.

All training entrypoints default to 500 optimizer updates, including extensions
and pilot/smoke profiles. Reduced datasets repeat for enough epochs to reach
this budget. Extensions retain their interaction budgets and runtime settings.
