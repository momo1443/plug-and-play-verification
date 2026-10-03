# Paper launcher configuration

The DeepScaleR paper launcher and the multi-turn ToolEnv launcher are different
experiment protocols. Keep their token allocations separate:

| Entry point | Profile | Prompt tokens | Response tokens |
| --- | --- | ---: | ---: |
| `scripts/deepscaler/run_deepscaler_a9_uniform.sh` | `deepscaler_paper` | 2048 initial; 8192 history | 4096 (4B) / 5120 (9B) total per trajectory |
| `scripts/extensions/deepscaler/run_deepscaler_tool_a9.sh` | `deepscaler_toolenv` | 4096 | 2048 per interaction step |

The main math flow defaults to `DEEPSCALER_MAX_STEPS=5`; `1` restores the
manuscript's original single-generation prompt, and larger positive integers
allow more reasoning turns. It uses a per-turn cap of `ceil(total / max_steps)`
and no answer-check feedback. Its total budget and context overrides are
`DEEPSCALER_MAX_RESPONSE_LENGTH` and `DEEPSCALER_MAX_MODEL_LENGTH`; apply the
same settings to training arms and AIME evaluation. Other domain profiles
retain their original turn counts and per-generation budgets. All main profiles
now default to prompt batch 20 for 4B and 8 for 9B, following the updated
eight-A40 hardware configuration. The manuscript's 9B hyperparameter tables
need this batch revision; 500 updates sample 4,000 prompts for 9B, while the
selected training prefix remains 10,000 rows.

ToolEnv A1 and A9 use the same shared launcher. Its length overrides are
`DEEPSCALER_TOOL_MAX_PROMPT_LENGTH`, `DEEPSCALER_TOOL_MAX_RESPONSE_LENGTH`, and
`DEEPSCALER_TOOL_MAX_MODEL_LENGTH` (context limit; default 8192). Changing lengths
does not convert the paper reward/interaction protocol into ToolEnv. The profile
and prompt/response limits are written to each DeepScaleR run manifest.

## Model adaptation

The DeepScaleR, HotpotQA, TACO A1/A9, and Vision-R1 launchers source
`model_training.sh`. They pass the same three model fields to verl: `lora_rank`,
`lora_alpha`, and `target_modules`, under `actor_rollout_ref.model`.

For a model path whose final component is `Qwen3.5-9B`, the default rank is 64.
Other model names default to rank 0 (full fine-tuning). The dedicated HotpotQA
and TACO 9B wrappers also set rank 64, including when the checkpoint was renamed.
LoRA alpha defaults to 64, with targets
`[q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj]`, matching the supplied
manuscript configuration.

Explicit environment overrides take precedence:

```bash
# Example for a renamed 9B checkpoint; retain the task's model-path setting.
AGENT_R1_LORA_RANK=64 AGENT_R1_LORA_ALPHA=64 bash scripts/hotpotqa/run_a9_uniform_9b.sh

# Reproduce the previous full-fine-tuning configuration explicitly.
AGENT_R1_LORA_RANK=0 bash scripts/hotpotqa/run_a9_uniform_9b.sh
```

Use `AGENT_R1_LORA_TARGET_MODULES='[q_proj,v_proj]'` to override the target list.
No new rollout-side LoRA flags are needed: upstream verl initializes vLLM LoRA
from the same model config. See the versioned
[model fields](https://github.com/volcengine/verl/blob/v0.7.0/verl/trainer/config/model/hf_model.yaml)
and [vLLM server implementation](https://github.com/volcengine/verl/blob/v0.7.0/verl/workers/rollout/vllm_rollout/vllm_async_server.py).
The same scalar fields remain in the
[verl 0.8.0 FSDP engine](https://github.com/volcengine/verl/blob/v0.8.0/verl/workers/engine/fsdp/transformer_impl.py),
and its vLLM server falls back to `lora_rank` when the nested LoRA rank is zero.

These are defaults for **new launches**, not evidence that previous 9B runs used
LoRA. For fair comparisons, set the same adaptation mode in all reward arms.
After configuration validation, the trainer writes an effective configuration
snapshot to `trainer.default_local_dir/launch_configs/resolved_<UTC>.yaml`.
Each launch, including a resume, has a separate timestamped file. Use that file
and the run manifest when preparing the final paper tables.

## Validation scope

Run `python3 -m unittest discover -s tests -p test_paper_launcher_config.py -v`
from the repository root. These tests run the actual shell launchers with a
Python stub, exercise model/length overrides, and capture trainer arguments.
They do not load models or run distributed optimization. The recorded earlier
FSDP2/offload/LoRA synchronization failure in `docs/archive/progress/2026-09-06.md` still
requires a GPU smoke test on the intended runtime before claiming stable LoRA
training. This change does not alter advantage, clipping, KL, or loss code.
