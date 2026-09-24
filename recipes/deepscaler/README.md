# DeepScaleR ToolEnv A1/A9

`run_deepscaler_tool_a1.sh` and `run_deepscaler_tool_a9.sh` are a paired
multi-turn experiment. Both use the same five-turn answer-check ToolEnv, model,
derived parquet, training order, seed, rollout count, sequence limits, and
runtime settings. The launcher rejects a sample/step mismatch so both arms see
the same 30,000 examples by default.

The pair differs only in its trajectory-level training reward:

| Arm | Training reward |
| --- | --- |
| A1 | Strict final-answer EM, placed on the final natural-language answer step. |
| A9 | A1 for the first 50 steps; afterwards `w * terminal_EM + (1 - w) * equation_process`, with one `w ~ Uniform(0, 1)` per trajectory. |

The equation process score excludes tool-call XML and tool observations. It
first averages verified/extracted numeric equalities within each parsed
reasoning step, then averages only turns with at least one extracted equality.
The full trajectory reward is placed on the final step so GRPO's
`step_causal` credit assignment can propagate it to earlier tool decisions.

The paired parquet is generated under `data/corpus/deepscaler_tool_pair/` and
intentionally omits `agent_name`; the launcher selects the A1 or A9 flow. Its
hidden answers remain in runner-side `env_kwargs`, never in the prompt.

Validation always reports strict terminal EM. The checker remains available
during validation, so this is an interactive agent metric and must be reported
as such. A no-checker transfer evaluation is a separate secondary protocol.
