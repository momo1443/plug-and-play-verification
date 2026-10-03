# Run entrypoints

Run these commands from the repository root, with the veRL environment, model
files, prepared datasets, and each task's services available. See the
[paper contract](../docs/paper-contract.md) for the matched profiles and
[shared settings](common/README.md) for 4B/9B model adaptation.

## Paper training

| Domain | Outcome-only GRPO | Verifier GRPO |
| --- | --- | --- |
| DeepScaleR | `bash scripts/deepscaler/run_deepscaler_a1.sh` | `bash scripts/deepscaler/run_deepscaler_a9_uniform.sh` |
| HotpotQA | `bash scripts/hotpotqa/run_a1.sh` | `bash scripts/hotpotqa/run_a9_uniform.sh` |
| TACO | `bash scripts/taco/run_a1_terminal.sh` | `bash scripts/taco/run_a9_uniform.sh` |
| Vision-R1 | `bash scripts/vision_r1/run_a1_terminal.sh` | `bash scripts/vision_r1/run_a9_uniform.sh` |

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
the manuscript's original single-turn setting. The total generated-token budget
remains 4096 (4B) or 5120 (9B); no correctness feedback enters this flow.

HotpotQA and TACO also provide `run_a9_uniform_9b.sh` wrappers. Domain launchers
share model adaptation from `common/model_training.sh`. `run_a7.sh` is the
retained weak-execution control; `run_a9.sh` is an alternative hardware wrapper
for the same HotpotQA verifier contract.

## Evaluation

| Task | Entry point |
| --- | --- |
| AIME 2025 | `scripts/deepscaler/eval_aime2025.py`; paired A0 wrapper `scripts/deepscaler/run_aime2025_a0.sh` |
| HotpotQA A0 | `scripts/hotpotqa/run_a0.sh` |
| HotpotQA certificate checkpoints | `scripts/hotpotqa/run_validation.sh` |
| MATH-Vision A0 | `scripts/mathvision/run_a0_react.sh`; standalone evaluator `recipes/mathvision/evaluate_react.py` |
| LiveCodeBench A0 | `recipes/livecodebench_a0/evaluate_a0.py` |
| Audit metrics from JSONL | `python -m agent_r1.evaluation.summarize validation.jsonl --output metrics.json` |

These are the existing evaluation utilities, not a claim that all paper scores
have been reproduced. Task validation and held-out benchmark evaluation remain
separate protocols.

## Extensions and environment patches

- `extensions/deepscaler/`: multi-turn answer-checking ToolEnv experiments. The incomplete DeepMath-103K launcher is preserved under `docs/archive/launchers/`; its `recipes/deepmath/` dependency was absent.
- `extensions/hotpotqa_lr/`: A8 local-reasoning training, calibration, validation, and diagnostics.
- `extensions/hotpotqa/`: old GPU/resume/storage-specific profiles. `run_a9_fixed.sh` still names a historical resume checkpoint; it is not a fresh-run default.
- `extensions/taco/`: one-step smoke tests and separated actor/rollout profiles.
- `patches/`: explicit environment patch scripts. Inspect their expected installed source/version before applying; moving them does not apply any patch.

Extensions retain their own budgets and runtime settings. Use the paper table
above for the matched experiments.
