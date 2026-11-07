# Run entrypoints

Run these commands from the repository root, with the veRL environment, model
files, prepared datasets, and each task's services available. See the
[training profiles](common/pnp.sh) for the domain settings and
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

Both arms of a domain go through the same model profile, so 4B/9B adaptation, data
preparation and memory settings are identical; only the reward differs. Add
`--model Qwen3.5-9B` for the 9B (LoRA) profile. Trailing `-- KEY=VALUE` Hydra overrides
reach the trainer in every domain.

| Domain | Outcome-only GRPO | Verifier GRPO |
| --- | --- | --- |
| DeepScaleR | `bash run_pnp.sh deepscaler --arm outcome` | `bash run_pnp.sh deepscaler` |
| HotpotQA | `bash run_pnp.sh hotpotqa --arm outcome` | `bash run_pnp.sh hotpotqa` |
| TACO | `bash run_pnp.sh taco --arm outcome` | `bash run_pnp.sh taco` |
| Vision-R1 | `bash run_pnp.sh vision_r1 --arm outcome` | `bash run_pnp.sh vision_r1` |

```bash
# Frozen process judge; domain: deepscaler, hotpotqa, taco, or vision.
bash scripts/llm_judge/run_grpo_4b_judge_9b.sh deepscaler
# Terminal-only PPO through the same domain interface.
bash scripts/ppo/run_terminal.sh deepscaler
# HotpotQA privileged gold-evidence ablations.
bash scripts/hotpotqa/run_a2.sh
bash scripts/hotpotqa/run_a3.sh
```

Math training defaults to at most five policy generations, shared by A1, A9,
Judge and PPO, including the last-turn boxed submission. Use
`DEEPSCALER_MAX_STEPS=8` for more turns or `=1` for a single generation.
The total training output budget is 4096 tokens for 4B and 5120 for 9B;
no correctness feedback enters this flow. Independent AIME evaluation uses
five generations with a 10240-token total budget and a 2048-token per-turn cap.

The shared `models/Qwen3.5-9B.sh DATASET` profile configures HotpotQA
and TACO 9B memory settings. Domain launchers share model adaptation from
`common/pnp.sh`. `run_a7.sh` is the
retained weak-execution control; `pnp_hotpotqa_8gpu.sh` is an alternative hardware wrapper
for the same HotpotQA verifier contract.

## Evaluation

Use `bash run_pnp_eval.sh BENCHMARK ...` for matched PNP evaluation of
`aime2025`, `aime2026`, `hotpotqa`, `livecodebench`, and `mathvision`.
The [shared evaluator](../agent_r1/evaluation/pnp.py) handles frozen dataset
revisions, model exports, final graders, and CPU preflight. Benchmark budgets
are defined in the [evaluation settings](../agent_r1/evaluation/pnp_protocol.py).
The entries below include compatibility wrappers and base-model baselines.

| Task | Entry point |
| --- | --- |
| AIME2025/AIME2026 PNP compatibility wrapper | `scripts/deepscaler/pnp_aime.py`, delegating to the shared evaluator |
| AIME 2025 A0 base-model baseline (use `run_pnp_eval.sh aime2025` or `aime2026` for the PNP protocol) | `scripts/deepscaler/eval_aime2025.py`; paired A0 wrapper `scripts/deepscaler/run_aime2025_a0.sh` |
| HotpotQA A0 | `scripts/hotpotqa/run_a0.sh` |
| HotpotQA PNP compatibility wrapper | `scripts/hotpotqa/pnp_hotpotqa_eval.sh`, delegating to the shared evaluator |
| MATH-Vision A0 | `scripts/mathvision/run_a0_react.sh`; standalone evaluator `recipes/mathvision/evaluate_react.py` |
| LiveCodeBench A0 | `recipes/livecodebench_a0/evaluate_a0.py` |
| Audit metrics from JSONL | `python -m agent_r1.evaluation.summarize validation.jsonl --output metrics.json` |

Use task validation during training and the held-out benchmark entrypoints for
final evaluation. Each run records its dataset, model, grader, and inference settings.

## Extensions and environment patches

- `extensions/deepscaler/`: multi-turn answer-checking ToolEnv experiments.
- `extensions/hotpotqa_lr/`: A8 local-reasoning training, calibration, validation, and diagnostics.
- `patches/`: explicit environment patch scripts. Inspect their expected installed source/version before applying; moving them does not apply any patch.

All training entrypoints default to 500 optimizer updates, including extensions
and pilot/smoke profiles. Reduced datasets repeat for enough epochs to reach
this budget. Extensions retain their interaction budgets and runtime settings.
