# DeepScaleR paper profile and ToolEnv extension

Paper experiments use `examples/deepmath/run_deepscaler_a1.sh` and
`run_deepscaler_a9_uniform.sh`: the same single-turn flow, 10,000 selected rows,
500 updates, 4 rollouts, and the common `VerificationResult` composer. The
frozen-judge launcher uses this same flow. Token limits are 2048/4096 for 4B and
2048/5120 for 9B. See [the paper contract](../../docs/paper-contract.md).

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
