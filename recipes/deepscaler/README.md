# DeepScaleR paper profile and ToolEnv extension

The main entrypoints are `bash run_pnp.sh deepscaler --arm outcome` and
`bash run_pnp.sh deepscaler`. Both select the first 10,000 rows of the original
training split and train for 500 updates, with 20 prompts per batch at both model
sizes (Appendix C) and four rollouts per prompt. Both use the shared
`VerificationResult` reward composer. The frozen-judge and PPO launchers use
the same interaction protocol and generation limits.
See the [shared training settings](../../scripts/common/README.md).

## Local Qwen3.5 setup

Use the activated Conda `pnp` environment for both installation and execution:

```bash
conda activate pnp
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
uv pip install --python "$PYTHON_BIN" \
  --override recipes/deepscaler/requirements-qwen35-overrides.txt \
  -r recipes/deepscaler/requirements-qwen35.txt
"$PYTHON_BIN" scripts/patches/patch_verl_lora_fsdp_checkpoint.py
# Place deepscaler.json under data/raw/DeepScaleR-Preview-Dataset/ first.
"$PYTHON_BIN" -m recipes.deepscaler.prepare_data
DEEPSCALER_TOTAL_TRAINING_STEPS=2 DEEPSCALER_SAVE_FREQ=-1 \
  bash scripts/deepscaler/pnp_deepscaler_smoke.sh
bash run_pnp.sh deepscaler --model Qwen3.5-4B --gpus 0,1,2,3,4,5,6,7
```

Use the recipe scripts from the activated Conda `pnp` environment:

```bash
DEEPSCALER_TOTAL_TRAINING_STEPS=2 DEEPSCALER_SAVE_FREQ=-1 \
  bash scripts/deepscaler/pnp_deepscaler_smoke.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/deepscaler/pnp_deepscaler.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  DEEPSCALER_RESUME_MODE=resume_path \
  DEEPSCALER_RESUME_FROM_PATH=/path/to/run/checkpoints/global_step_500 \
  DEEPSCALER_VAL_ONLY=True DEEPSCALER_VAL_BEFORE_TRAIN=True \
  bash scripts/deepscaler/pnp_deepscaler.sh
```

Run artifacts, TensorBoard records, validation JSONL and Hydra metadata are
stored under project-local `logs/`. Periodic validation is disabled by default
(`DEEPSCALER_TEST_FREQ=-1`), including evaluation at the final training step.
Training-trajectory verification continues to run. `DEEPSCALER_TEST_FREQ` and
`DEEPSCALER_VAL_BEFORE_TRAIN` configure optional periodic and initial validation.
To validate a saved FSDP checkpoint, set `DEEPSCALER_RESUME_MODE=resume_path`
and `DEEPSCALER_RESUME_FROM_PATH=/path/global_step_500` on the validation command.
Validation scores final-answer correctness and audits the equation verifier
separately. The GSM8K upstream tool tutorial is a different interactive
protocol: its answer-checking tool feeds reference-based correctness back to
the model. The paper mathematics flow here provides fixed continuation
messages and performs verification after the rollout.

The main model profile validates the token lengths of the original 10,000-row
training prefix before launch. Rich verifier audits are retained in validation
JSONL; aggregate validation metrics use numeric scalar fields.

The `scripts/models/Qwen3.5-4B.sh` profile, selected through `run_pnp.sh`, selects exactly
the first 10,000 original training rows. Overlong prefix rows raise an error;
rows are never replaced by later examples and reference labels remain unchanged.
It defaults to eight NVIDIA A40 GPUs (0–7), a batch and mini-batch size of 20 prompts, four trajectories per prompt,
500 updates in one epoch, AdamW at constant 1e-6 with zero LR warmup, weight decay
0.01, gradient clipping 1.0, KL 0.001, bfloat16 and seed 42. Its generation limits
are 2,048 initial tokens, 8,192 context tokens, and 4,096 completion tokens across
up to five turns, capped at 820 tokens per turn and the remaining total budget.
The 9B profile uses 5,120 completion tokens with a 1,024-token per-turn cap.
The 4B model profile saves checkpoints every 10 updates;
periodic held-out validation is disabled. The default A9 outcome-only reward
warmup lasts 50 updates and is independent of learning-rate warmup.
Set `DEEPSCALER_EM_WARMUP_STEPS=0` to disable reward warmup.
The prepared subset, its source-selection manifest and resolved configuration
are saved in the run's `logs/` directory.

The model profile uses a vLLM memory fraction of 0.30, maximum concurrent
sequences of 32, and 16 AgentFlow workers on eight GPUs. These settings accept
the usual `DEEPSCALER_VLLM_*` overrides.
GPU utilization measures kernel activity; GPU memory usage is a separate metric.
Size the vLLM budget from the complete training peak, including actor backward
and optimizer states, rather than from generation-phase memory alone.

The default uses the original training split without cleaning or label changes.
Optional `DEEPSCALER_CLEAN_DATA=True` enables a separate cleaning experiment before
prefix selection. In that mode every source row and distinct reference label is checked;
non-equivalent labels for duplicate questions, missing referenced diagrams and
unsupported answer formats are quarantined. The independently checked dress
discount/tax reference is corrected from 54 to 66. Original source files and
held-out membership are preserved; the cleaned splits and full audit are saved
under `_meta/data/`. Explicit `DEEPSCALER_TRAIN_PATH` / `DEEPSCALER_VAL_PATH`
overrides select supplied data. `DEEPSCALER_CLEAN_DATA=False` (the default) selects the original
corpus for comparisons. Cleaning reports record structural checks, duplicate-label
comparisons, exclusions, and reference corrections.

The local profile disables Ray OpenTelemetry and sets
`VERL_FSDP_CHECKPOINT_OFFLOAD_TO_CPU=0` to avoid assembling large CPU copies
during checkpoint saves. Apply the installed veRL checkpoint compatibility patch
with `python scripts/patches/patch_verl_lora_fsdp_checkpoint.py` before using this
setting. It changes checkpoint staging, while retaining model, optimizer and
training-state contents. `VERL_FSDP_CHECKPOINT_OFFLOAD_TO_CPU=1` restores CPU staging.

The numeric verifier preserves thousands separators, nested/unbraced fractions,
mixed numbers, indexed roots, explicit repeating decimals and numeric bases.
Rational arithmetic is exact, and floating transcendental functions have a
small numerical tolerance. Symbolic expressions, ambiguous rounding/modular/
unit-conversion contexts and custom star operations earn no process credit.
Refuted equalities and running-total shorthand are excluded. True arithmetic
mistakes remain failed checks. Final answers are compared as selected math
expressions, including dollar wrappers and equivalent fraction formatting.
The evidence parser preserves operators across inline math/currency delimiters
and supports rational generalized binomials and multinomials. Numbered labels,
list bullets, percentage display shorthand and rounded decimal displays cannot
create spurious arithmetic failures or earn exact-equality credit. Length-capped
generations record `truncated_policy_steps`; only the unfinished final equality
is excluded, while preceding complete equalities remain checked. Selected
monomial answers are compared exactly without symbolic subprocess startup;
equations remain distinct from answers consisting only of their roots.
Thinking text and quoted format examples are not final submissions. When a
Qwen generation prompt opens `<think>`, a closing `</think>` is required before
a submitted answer. Reasoning turns use the model's default thinking mode.
The last available turn is a concise visible answer submission with
`enable_thinking=False`, using the same remaining completion budget (normally
816 tokens after four 820-token turns for 4B; 1,024 for 9B).
Standalone AIME2025/AIME2026 evaluation uses the same turn protocol with its own
10,240-token budget (2,048 per turn, 13,312 context). Set `DEEPSCALER_FINAL_TURN_SUBMISSION=False`
to reproduce the full-thinking protocol. All token budgets remain unchanged.
The final submission prompt prefills `\boxed{` as prompt tokens, then the model
generates the answer and its closing brace. Nested and escaped math braces are
handled when locating the first complete answer. Training generates this turn
in chunks of at most 64 tokens and stops requesting chunks once a complete box
is available; transport chunks still constitute one policy turn. Original
sampled token IDs and log probabilities are retained through the closing box,
and any generated suffix discarded from policy training still counts against
the trajectory's total completion budget. `final_answer_prefill_step` records the turn for offline
replay; rollout output contains the generated suffix, with the prefix in its
input. Submission-only output does not receive process credit.
Earlier continuation prompts advance the existing solution and request a
recheck only for a specific inconsistency. The penultimate turn prepares a
candidate for the final submission. These prompts contain no verifier feedback
or reference answer, and all math reward arms use the same protocol.
Symbolic answer comparisons in Ray async actor threads use an isolated CPU
process, retaining parser/comparison deadlines and an overall eight-second
deadline. This avoids signal-based timeouts incorrectly rejecting valid
equivalent answers outside the interpreter main thread.

Full offline replay is available with:

```bash
python -m recipes.deepscaler.audit_rollouts \
  --input /path/to/run/rollouts.jsonl \
  --quality-report /path/to/new/run/_meta/data/quality_audit.json \
  --output /path/to/new/run/_meta/full_rollout_audit.json
```

Replay computes parser and verifier diagnostics from the recorded trajectory text.
The optional quality report supplies reference corrections and exclusion records
for that audit; omit `--quality-report` to use the original source references.

The Qwen3.5 runtime uses veRL 0.8.0, which supplies the `qwen3_5` training
adapter absent from the released 0.7.0 wheel. The override file explicitly
resolves vLLM 0.17.1's older Transformers upper bound and the NumPy/OpenCV
constraint conflict; it is specific to this text-only math profile.
The launcher enables `AGENT_R1_TORCH_PADDING=1`: when FlashAttention is absent,
token packing uses veRL's existing device-independent PyTorch padding helpers.
The model still uses SDPA. This avoids requiring a CUDA extension solely for
padding; installing a working FlashAttention package takes precedence.

Data conversion reads the local DeepScaleR JSON, holds out at least 2,015 rows
using a seeded shuffle of question groups, and retains source order within each
split. Repeated questions stay together to avoid train/validation leakage.
Rows missing problems or answers are excluded and recorded in
`data/corpus/deepscaler/data_manifest.json`. Source answer conflicts are counted
and retained without changing labels. This creates a local diagnostic split;
it does not substitute for the AIME 2025/2026 evaluation sets.

The smoke profile defaults to eight GPUs, a batch of four training examples,
two rollouts per prompt, a 512-token trajectory budget, validation every update,
and zero reward warmup. Its default update count is 500 and its checkpoint
interval is inherited from the main launcher (50). The two-update examples above
set `DEEPSCALER_TOTAL_TRAINING_STEPS=2` and `DEEPSCALER_SAVE_FREQ=-1` explicitly.
The main launcher uses the 50-update warmup and the paper's generation budget. Select free GPUs
explicitly when launching the main profile; its default expects eight GPUs.

The main launcher accepts `DEEPSCALER_MODEL_PATH`, `DEEPSCALER_TRAIN_PATH`,
`DEEPSCALER_VAL_PATH`, `DEEPSCALER_TRAIN_BATCH_SIZE`, `DEEPSCALER_ROLLOUT_N`,
`DEEPSCALER_VAL_MAX_SAMPLES`, `DEEPSCALER_VAL_BATCH_SIZE`,
`DEEPSCALER_VLLM_MAX_NUM_SEQS`, `DEEPSCALER_VLLM_GPU_MEMORY_UTILIZATION`, and
`DEEPSCALER_OUTPUT_DIR`. Additional Hydra arguments can follow the script name.

The mathematics results reported in the manuscript use at most five policy
generations. The model receives a fixed instruction to continue or recheck its reasoning; it receives no hidden
answer, correctness signal, or process-verifier feedback. A parseable final
answer ends the trajectory even when incorrect. Process checks run after the
trajectory ends, with credits backfilled to their reasoning turns and the
outcome reward placed at the final policy step. If the turn, token, or context
limit is reached without a final answer, all composed task rewards are zero;
recorded steps, policy-token masks, and diagnostic audits are retained.

For each policy turn, the extractor checks numeric equations in its reasoning
text, excluding the selected final answer. Each equation audit records
`source_step`; internal reasoning-segment indices remain separate. The local
turn score `z_t` is the mean verified-equation fraction over its equation-bearing
internal segments, preserving the original scoring rule.

Let `I` contain **all recorded policy turns with at least one extractable numeric
equation**, including turns whose checks all fail. The adapter returns
`p_t = z_t / len(I)` for these turns and zero for the others, so
`P = sum(p_t) <= 1`. A turn without extractable equations does not participate
in this denominator and receives zero credit; when `I` is empty, `P=0`.
Unsupported expressions are excluded from credit; supported arithmetic with
invalid numeric domains fails its checks. The audit
records the participating turns, normalizer, and each normalized credit. Missing
reasoning records or duplicate source-step indices raise an explicit adapter
error rather than silently dropping a required record.
The parser-version audit includes reasons for explicitly excluded rounded,
percentage, variable-pi and incomplete-tail evidence. Such exclusions do not
count as passed checks and are outside the extractable-equation denominator.

A corrected equation is a new check at its own source turn. An earlier failed
check remains in `applicable_checks`: for example, a failed first turn followed
by a fully verified second turn yields credits `0` and `1/2`, while strict
trajectory-level consistency remains zero. This distinguishes successful correction
from a history in which every applicable check passed.

The observation between generations is the fixed continuation/recheck prompt
from `prompts.py`. It contains no verifier results or reference-answer signal.
Both deterministic and frozen-judge verification run after rollout completion;
outcome scoring uses the reference only after a final answer has been selected.

The original input limit is 2048 tokens. Multi-turn training history uses an 8,192-token
context by default (AIME evaluation uses 13,312). **The total generated-token budget per trajectory** is
4,096 for 4B and 5,120 for 9B, with a per-turn cap of `ceil(total / max_steps)`.
For five turns, these caps are 820 and 1,024 respectively; each generation is also capped by
the remaining trajectory budget and context space.
Unused tokens from an early-stopped turn are not added to a later turn's cap.
Generation ends on a final submission or an exhausted turn, output-token, or
context budget. Recorded history is preserved when context space is exhausted.

```bash
# Default: at most five turns; the training option applies to A1, A9, Judge and PPO.
DEEPSCALER_MAX_STEPS=5 bash scripts/deepscaler/pnp_deepscaler.sh
# More turns, retaining the same total token budget.
DEEPSCALER_MAX_STEPS=8 bash scripts/deepscaler/pnp_deepscaler.sh
# Optional single-generation setting.
DEEPSCALER_MAX_STEPS=1 bash scripts/deepscaler/pnp_deepscaler.sh
```

Optional budget/context overrides are `DEEPSCALER_MAX_RESPONSE_LENGTH` and
`DEEPSCALER_MAX_MODEL_LENGTH`; use identical values in all comparison arms, and
pass the same values to evaluation (`--total-output-tokens`/`--context-limit`, or
`AIME2025_MAX_RESPONSE_LENGTH`/`AIME2025_MAX_MODEL_LENGTH` for `eval_aime2025.py`). Run manifests record the effective
turn cap, total/per-turn budgets, context length, and prompt code hashes.
The single-generation and eight-generation settings are optional alternatives
to the five-generation protocol used for the reported mathematics results.

## Optional ToolEnv extension

`scripts/extensions/deepscaler/run_deepscaler_tool_a1.sh` and
`scripts/extensions/deepscaler/run_deepscaler_tool_a9.sh` are a paired
multi-turn experiment. Both use the same five-turn answer-check ToolEnv, model,
derived parquet, training order, seed, rollout count, sequence limits, and
runtime settings. The launcher rejects a sample/step mismatch so both arms see
the same 10,000 examples by default (20 prompts per update for 500 updates).

The pair differs only in its trajectory-level training reward:

| Arm | Training reward |
| --- | --- |
| A1 | Strict final-answer EM, placed on the final natural-language answer step. |
| A9 | A1 for the first 50 steps; afterwards `w * terminal_EM + (1 - w) * equation_process`, with one `w ~ Uniform(0, 1)` per prompt group/update. |

The equation process score excludes tool-call XML and tool observations. It
first averages verified/extracted numeric equalities within each parsed
reasoning step, then averages only turns with at least one extracted equality.
Process credits remain on their source turns; terminal EM is placed at the final
step. `step_causal` GRPO computes causal returns before one group normalization.
Missing final submissions receive zero composed reward.

The paired parquet is generated under `data/corpus/deepscaler_tool_pair/` and
intentionally omits `agent_name`; the launcher selects the A1 or A9 flow. Its
hidden answers remain in runner-side `env_kwargs`, never in the prompt.

Validation always reports strict terminal EM. The checker remains available
during validation, so this is an interactive agent metric and must be reported
as such. A no-checker transfer evaluation is a separate secondary protocol.
