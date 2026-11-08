# Paper launcher configuration

Training launchers default to 500 optimizer updates, including extension and
pilot/smoke launchers. `pnp.sh` defines this default and computes enough training
epochs for reduced datasets to reach the requested update count.
The trainer YAML also defaults to 500 steps for direct Python launches.

The DeepScaleR paper launcher and the multi-turn ToolEnv launcher are different
experiment protocols. Keep their token allocations separate:

| Entry point | Profile | Prompt tokens | Response tokens |
| --- | --- | ---: | ---: |
| `scripts/deepscaler/pnp_deepscaler.sh` | `deepscaler_paper` | 2048 initial; 8192 history | 4096 total (4B) / 5120 total (9B) |
| `scripts/extensions/deepscaler/run_deepscaler_tool_a9.sh` | `deepscaler_toolenv` | 4096 | 2048 per interaction step |

The main math flow defaults to `DEEPSCALER_MAX_STEPS=5`; `1` selects an optional single-generation setting, and
larger positive integers allow more reasoning turns. It uses a per-turn cap of
`ceil(total / max_steps)` and no answer-check feedback. Its total budget and context overrides are
`DEEPSCALER_MAX_RESPONSE_LENGTH` and `DEEPSCALER_MAX_MODEL_LENGTH`; apply the
same training settings to all reward arms. Independent AIME evaluation uses
10,240 output tokens with a 2,048-token per-turn cap. Other domain profiles
retain their original turn counts and per-generation budgets. Appendix C prompt
and mini-batches are 20 for DeepScaleR and TACO at both model sizes, 20/8 for
HotpotQA 4B/9B, and 8 for Vision-R1 at both model sizes. At 500 updates, batch
20 samples 10,000 prompts and batch 8 samples 4,000 prompts; the selected
training prefix remains 10,000 rows.

ToolEnv A1 and A9 use a shared launcher. Its length overrides are
`DEEPSCALER_TOOL_MAX_PROMPT_LENGTH`, `DEEPSCALER_TOOL_MAX_RESPONSE_LENGTH`, and
`DEEPSCALER_TOOL_MAX_MODEL_LENGTH` (context limit; default 8192). Changing lengths
does not convert the paper reward/interaction protocol into ToolEnv. The profile
and prompt/response limits are written to each DeepScaleR run manifest.

## Model adaptation

The DeepScaleR, HotpotQA, TACO A1/A9, and Vision-R1 launchers source
`pnp.sh`. They pass the same three model fields to veRL: `lora_rank`,
`lora_alpha`, and `target_modules`, under `actor_rollout_ref.model`.

For a model path whose final component is `Qwen3.5-9B`, the default rank is 64.
Other model names default to rank 0 (full fine-tuning). The `Qwen3.5-9B.sh`
profile also sets rank 64, including when the checkpoint was renamed.
LoRA alpha defaults to 64, with targets
`[q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj]`, matching the supplied
manuscript configuration.

Explicit environment overrides take precedence:

```bash
# Example for a renamed 9B checkpoint; retain the task's model-path setting.
AGENT_R1_LORA_RANK=64 AGENT_R1_LORA_ALPHA=64 bash run_pnp.sh hotpotqa --model Qwen3.5-9B

# Select full fine-tuning explicitly.
AGENT_R1_LORA_RANK=0 bash run_pnp.sh hotpotqa --model Qwen3.5-9B
```

Use `AGENT_R1_LORA_TARGET_MODULES='[q_proj,v_proj]'` to override the target list.
veRL initializes rollout-side LoRA in vLLM from the same model configuration.
See the versioned
[model fields](https://github.com/volcengine/verl/blob/v0.7.0/verl/trainer/config/model/hf_model.yaml)
and [vLLM server implementation](https://github.com/volcengine/verl/blob/v0.7.0/verl/workers/rollout/vllm_rollout/vllm_async_server.py).
These fields are also used by the
[verl 0.8.0 FSDP engine](https://github.com/volcengine/verl/blob/v0.8.0/verl/workers/engine/fsdp/transformer_impl.py),
and its vLLM server falls back to `lora_rank` when the nested LoRA rank is zero.

Use the same adaptation mode in all reward arms for matched comparisons.
After configuration validation, the trainer writes an effective configuration
snapshot to `trainer.default_local_dir/launch_configs/resolved_<UTC>.yaml`.
Each launch, including a resume, has a separate timestamped file. Use that file
and the run manifest when preparing the final paper tables.

## Validation scope

Run `python3 -m unittest discover -s tests -p test_pnp_config.py -v`
from the repository root. These tests run the actual shell launchers with a
Python stub, exercise model/length overrides, and capture trainer arguments.
Use GPU smoke runs to check FSDP/LoRA execution and checkpoint synchronization
on the target runtime. Advantage computation, clipping, KL regularization,
and loss aggregation are defined in the shared trainer implementation.
