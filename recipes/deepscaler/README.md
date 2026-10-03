# DeepScaleR paper profile and ToolEnv extension

The main entrypoints are `scripts/deepscaler/run_deepscaler_a1.sh` and
`run_deepscaler_a9_uniform.sh`. Both use 10,000 selected rows, 500 updates,
prompt batch 20 (4B) or 8 (9B), four rollouts, and the common `VerificationResult` composer.
The frozen-judge and PPO launchers use the same flow and generation limits.
See [the implementation contract](../../docs/paper-contract.md).

Math now defaults to at most five policy generations. The model receives a
fixed instruction to continue or recheck its reasoning; it receives no hidden
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
Extracted equations with unsupported expressions fail their checks. The audit
records the participating turns, normalizer, and each normalized credit. Missing
reasoning records or duplicate source-step indices raise an explicit adapter
error rather than silently dropping a required record.

A corrected equation is a new check at its own source turn. An earlier failed
check remains in `applicable_checks`: for example, a failed first turn followed
by a fully verified second turn yields credits `0` and `1/2`, while strict
historical consistency remains zero. This distinguishes successful correction
from a history in which every applicable check passed.

The observation between generations is the fixed continuation/recheck prompt
from `prompts.py`. It contains no verifier results or reference-answer signal.
Both deterministic and frozen-judge verification run after rollout completion;
outcome scoring uses the reference only after a final answer has been selected.

The original input limit is 2048 tokens. Multi-turn history uses an 8192-token
context by default. **The total generated-token budget per trajectory** remains
4096 for 4B or 5120 for 9B, with a per-turn cap of `ceil(total / max_steps)`.
For five turns, these caps are 820 and 1024; each generation is also capped by
the remaining trajectory budget and context space.
Unused tokens from an early-stopped turn are not added to a later turn's cap.
These caps bound generation; they do not guarantee that every trajectory uses
all turns or all tokens. Context exhaustion ends the rollout without truncating
its recorded history.

```bash
# Default: at most five turns; same option applies to A1, Judge, PPO and AIME.
DEEPSCALER_MAX_STEPS=5 bash scripts/deepscaler/run_deepscaler_a9_uniform.sh
# More turns, retaining the same total token budget.
DEEPSCALER_MAX_STEPS=8 bash scripts/deepscaler/run_deepscaler_a9_uniform.sh
# Original single-generation prompt and completion limit.
DEEPSCALER_MAX_STEPS=1 bash scripts/deepscaler/run_deepscaler_a9_uniform.sh
```

Optional budget/context overrides are `DEEPSCALER_MAX_RESPONSE_LENGTH` and
`DEEPSCALER_MAX_MODEL_LENGTH`; use identical values in all comparison arms and
in `scripts/deepscaler/eval_aime2025.py`. Run manifests record the effective
turn cap, total/per-turn budgets, context length, and prompt code hashes.
The supplied manuscript describes single-turn math; its description and
reported math scores must be updated after evaluating this new protocol.

## Optional ToolEnv extension

`run_deepscaler_tool_a1.sh` and `run_deepscaler_tool_a9.sh` are a paired
multi-turn experiment. Both use the same five-turn answer-check ToolEnv, model,
derived parquet, training order, seed, rollout count, sequence limits, and
runtime settings. The launcher rejects a sample/step mismatch so both arms see
the same 30,000 examples by default.

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
