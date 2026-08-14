#!/usr/bin/env bash
set -euo pipefail
set -x

if (( $# != 0 )); then
    echo "This launcher takes no arguments; configure it with environment variables" >&2
    exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

[[ "${HOTPOTQA_VALIDATION_INTERFACE:-lr}" == "lr" ]] || {
    echo "A8-LR validation requires HOTPOTQA_VALIDATION_INTERFACE=lr" >&2
    exit 2
}

export HOTPOTQA_FORMAL_A0=0
export HOTPOTQA_FORMAL_EXPERIMENT=1
export HOTPOTQA_LR_REASON_STEP_FORMAT="${HOTPOTQA_LR_REASON_STEP_FORMAT:-dsl}"
case "$HOTPOTQA_LR_REASON_STEP_FORMAT" in
    dsl)
        export HOTPOTQA_REWARD_ARM=A8_LR30
        ;;
    claim_source)
        export HOTPOTQA_REWARD_ARM=A8_LR30_CS
        ;;
    *)
        echo "HOTPOTQA_LR_REASON_STEP_FORMAT must be dsl or claim_source" >&2
        exit 2
        ;;
esac
export HOTPOTQA_VALIDATION_INTERFACE=lr
export HOTPOTQA_ENABLE_THINKING=false
export HOTPOTQA_FORCE_FIRST_SEARCH=false
export HOTPOTQA_REQUIRE_SENTENCE_EVIDENCE=true
export HOTPOTQA_EMBEDDING_PER_WORKER_GPU=0
export HOTPOTQA_EMBEDDING_DEVICE=cpu
export HOTPOTQA_STREAMING_RESULTS=1
export HOTPOTQA_STREAMING_FSYNC=1
# LR exposes structured audit dictionaries. Aggregate only scalar task metrics;
# the complete ledgers remain in the per-sample JSONL records.
export HOTPOTQA_STREAMING_METRIC_KEYS=acc,terminal_em
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HOTPOTQA_DATA_ROOT="${HOTPOTQA_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa}"
export HOTPOTQA_CORPUS_DATA_ROOT="${HOTPOTQA_CORPUS_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa_corpus}"
export HOTPOTQA_EVIDENCE_SIDECAR="${HOTPOTQA_EVIDENCE_SIDECAR:-$HOTPOTQA_CORPUS_DATA_ROOT/hotpotqa_evidence_v1.sqlite3}"
export HOTPOTQA_EMBEDDING_MODEL="${HOTPOTQA_EMBEDDING_MODEL:-$WORKSPACE_DIR/models/bge-large-en-v1.5}"

MODEL_PATH="${HOTPOTQA_MODEL_PATH:?HOTPOTQA_MODEL_PATH is required}"
TRAIN_PATH="${HOTPOTQA_TRAIN_PATH:-$HOTPOTQA_DATA_ROOT/train.parquet}"
VAL_PATH="${HOTPOTQA_VAL_PATH:-$HOTPOTQA_DATA_ROOT/validation.parquet}"
AGENT_CONFIG="$PROJECT_DIR/recipes/hotpotqa_lr/base.yaml"
REWARD_PATH="$PROJECT_DIR/recipes/hotpotqa_lr/reward_fn.py"
SOURCE_CHECKPOINT="${HOTPOTQA_SOURCE_CHECKPOINT:-}"

VAL_MAX_SAMPLES="${HOTPOTQA_VAL_MAX_SAMPLES:--1}"
VAL_BATCH_SIZE="${HOTPOTQA_VAL_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH_PER_STEP=1024
MAX_MODEL_LENGTH=12288
IFS=',' read -r -a CUDA_DEVICE_LIST <<<"$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${HOTPOTQA_NUM_GPUS:-${#CUDA_DEVICE_LIST[@]}}"
TRAIN_BATCH_SIZE="${HOTPOTQA_TRAIN_BATCH_SIZE:-$(( (8 + NUM_GPUS - 1) / NUM_GPUS * NUM_GPUS ))}"
AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-$NUM_GPUS}"
VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
VLLM_MAX_NUM_SEQS="${HOTPOTQA_VLLM_MAX_NUM_SEQS:-8}"
VLLM_ENABLE_SLEEP_MODE="${HOTPOTQA_VLLM_ENABLE_SLEEP_MODE:-false}"
VLLM_FREE_CACHE_ENGINE="${HOTPOTQA_VLLM_FREE_CACHE_ENGINE:-false}"
VAL_N="${HOTPOTQA_VAL_N:-1}"
VAL_DO_SAMPLE="${HOTPOTQA_VAL_DO_SAMPLE:-false}"
VAL_TEMPERATURE="${HOTPOTQA_VAL_TEMPERATURE:-0}"
VAL_TOP_P="${HOTPOTQA_VAL_TOP_P:-1}"
VAL_TOP_K="${HOTPOTQA_VAL_TOP_K:--1}"
CONFIG_ONLY="${HOTPOTQA_HYDRA_CONFIG_ONLY:-0}"

if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || (( NUM_GPUS != ${#CUDA_DEVICE_LIST[@]} )); then
    echo "HOTPOTQA_NUM_GPUS must match CUDA_VISIBLE_DEVICES" >&2
    exit 2
fi
if (( TRAIN_BATCH_SIZE <= 0 || TRAIN_BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "HOTPOTQA_TRAIN_BATCH_SIZE must be positive and divisible by the GPU count" >&2
    exit 2
fi
if [[ "$VAL_N" != "1" || "${VAL_DO_SAMPLE,,}" != "false" || "$VAL_TEMPERATURE" != "0" ]]; then
    echo "Formal A8-LR validation requires n=1, do_sample=false, temperature=0" >&2
    exit 2
fi
case "$CONFIG_ONLY" in
    0|1) ;;
    *) echo "HOTPOTQA_HYDRA_CONFIG_ONLY must be 0 or 1" >&2; exit 2 ;;
esac
for bool_name in VLLM_ENABLE_SLEEP_MODE VLLM_FREE_CACHE_ENGINE; do
    case "${!bool_name}" in
        true|false) ;;
        *) echo "$bool_name must be true or false" >&2; exit 2 ;;
    esac
done
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing model config: $MODEL_PATH" >&2; exit 2; }
find "$MODEL_PATH" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q . || {
    echo "Missing model safetensors: $MODEL_PATH" >&2
    exit 2
}

RUN_ID="${RUN_ID:-qwen35-4b_a8-lr30_native_full7405_${NUM_GPUS}gpu_$(date +%Y%m%d-%H%M%S)}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ar1-lrval-$$}"
mkdir -p "$VALIDATION_DATA_DIR"
cd "$PROJECT_DIR"

RUN_ID="$RUN_ID" \
VALIDATION_DATA_DIR="$VALIDATION_DATA_DIR" \
MODEL_PATH="$MODEL_PATH" \
SOURCE_CHECKPOINT="$SOURCE_CHECKPOINT" \
TRAIN_PATH="$TRAIN_PATH" \
VAL_PATH="$VAL_PATH" \
VAL_MAX_SAMPLES="$VAL_MAX_SAMPLES" \
VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
MAX_RESPONSE_LENGTH_PER_STEP="$MAX_RESPONSE_LENGTH_PER_STEP" \
MAX_MODEL_LENGTH="$MAX_MODEL_LENGTH" \
VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
VLLM_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS" \
VLLM_ENABLE_SLEEP_MODE="$VLLM_ENABLE_SLEEP_MODE" \
VLLM_FREE_CACHE_ENGINE="$VLLM_FREE_CACHE_ENGINE" \
NUM_GPUS="$NUM_GPUS" \
AGENT_WORKERS="$AGENT_WORKERS" \
PROJECT_DIR="$PROJECT_DIR" \
"$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from recipes.hotpotqa.prepare_formal_rlvr_run import _model_identity, _package_versions
from recipes.hotpotqa_lr.dsl import DSL_VERSION, REASON_STEP_FORMAT
from recipes.hotpotqa_lr.protocol import LR_FINISH_PROTOCOL
from recipes.hotpotqa_lr.reward_contract import LR_CONTRACT_VERSION, LR_REWARD_ARM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


project_dir = Path(os.environ["PROJECT_DIR"])
output_dir = Path(os.environ["VALIDATION_DATA_DIR"])
model_path = Path(os.environ["MODEL_PATH"])
train_path = Path(os.environ["TRAIN_PATH"])
val_path = Path(os.environ["VAL_PATH"])
val_rows = int(pq.read_metadata(val_path).num_rows)
requested = int(os.environ["VAL_MAX_SAMPLES"])
selected = val_rows if requested < 0 else min(requested, val_rows)
if selected != 7_405:
    raise ValueError(f"A8-LR full validation requires 7,405 rows, got {selected}/{val_rows}")

code_paths = [
    "recipes/hotpotqa_lr/agent_flow.py",
    "recipes/hotpotqa_lr/base.yaml",
    "recipes/hotpotqa_lr/dsl.py",
    "recipes/hotpotqa_lr/prompts.py",
    "recipes/hotpotqa_lr/protocol.py",
    "recipes/hotpotqa_lr/reward_contract.py",
    "recipes/hotpotqa_lr/reward_fn.py",
    "recipes/hotpotqa_lr/verifier.py",
    "agent_r1/trainer/streaming_agent_validation.py",
    "agent_r1/trainer/compact_validation_record.py",
    "examples/hotpotqa_lr/run_validation.sh",
]
manifest = {
    "contract_version": "hotpotqa-a8-lr-native-validation-v1",
    "status": "prepared",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": os.environ["RUN_ID"],
    "arm": LR_REWARD_ARM.replace("_", "-"),
    "run_mode": "validation_only",
    "validation_interface": "lr_native",
    "final_answer_protocol": LR_FINISH_PROTOCOL,
    "reward_contract": {
        "contract_id": LR_CONTRACT_VERSION,
        "dsl_version": DSL_VERSION,
        "reason_step_format": REASON_STEP_FORMAT,
    },
    "checkpoint": os.environ.get("SOURCE_CHECKPOINT", ""),
    "output_dir": str(output_dir.resolve()),
    "output_jsonl": str((output_dir / "0.jsonl").resolve()),
    "model": _model_identity(model_path),
    "resources": {
        "cuda_visible_devices": [x.strip() for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")],
        "num_gpus": int(os.environ["NUM_GPUS"]),
        "agent_workers": int(os.environ["AGENT_WORKERS"]),
        "vllm_gpu_memory_utilization": float(os.environ["VLLM_GPU_MEMORY_UTILIZATION"]),
        "vllm_max_num_seqs": int(os.environ["VLLM_MAX_NUM_SEQS"]),
        "vllm_load_format": "auto",
        "initial_weight_sync": "skipped_source_loaded",
        "vllm_enable_sleep_mode": os.environ["VLLM_ENABLE_SLEEP_MODE"] == "true",
        "vllm_free_cache_engine": os.environ["VLLM_FREE_CACHE_ENGINE"] == "true",
        "val_batch_size": int(os.environ["VAL_BATCH_SIZE"]),
    },
    "scientific_config": {
        "validation_only": True,
        "optimizer_updates": 0,
        "validation_n": 1,
        "validation_do_sample": False,
        "validation_temperature": 0.0,
        "thinking_mode": "disabled",
        "max_agent_flow_turns": 4,
        "max_searches": 3,
        "terminal_metric": "normalized exact match",
        "process_reward_in_validation": False,
    },
    "selection": {
        "validation": {
            "path": str(val_path.resolve()),
            "sha256": sha256(val_path),
            "source_rows": val_rows,
            "count": selected,
        },
        "train_placeholder": {
            "path": str(train_path.resolve()),
            "count": int(os.environ.get("TRAIN_BATCH_SIZE", "8")),
            "optimizer_updates": 0,
        },
    },
    "code_sha256": {relative: sha256(project_dir / relative) for relative in code_paths},
    "package_versions": _package_versions(),
}
tmp = output_dir / ".run_manifest.json.tmp"
tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, output_dir / "run_manifest.json")
print(f"wrote native A8-LR validation manifest -> {output_dir / 'run_manifest.json'}")
PY

HYDRA_CONFIG_ARGS=()
if [[ "$CONFIG_ONLY" == "1" ]]; then
    HYDRA_CONFIG_ARGS=(--cfg job)
fi

"$PYTHON_BIN" -m agent_r1.trainer.main_agent_ppo \
    "${HYDRA_CONFIG_ARGS[@]}" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.train_max_samples="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.val_max_samples="$VAL_MAX_SAMPLES" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH_PER_STEP" \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
    +actor_rollout_ref.rollout.enable_sleep_mode="$VLLM_ENABLE_SLEEP_MODE" \
    actor_rollout_ref.rollout.free_cache_engine="$VLLM_FREE_CACHE_ENGINE" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LENGTH" \
    actor_rollout_ref.rollout.max_num_seqs="$VLLM_MAX_NUM_SEQS" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$AGENT_CONFIG" \
    actor_rollout_ref.rollout.agent.default_agent_flow=hotpotqa_local_reasoning_agent \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.top_k="$VAL_TOP_K" \
    critic.enable=False \
    reward_model.enable=False \
    custom_reward_function.path="$REWARD_PATH" \
    custom_reward_function.name=compute_score \
    reward.custom_reward_function.path="$REWARD_PATH" \
    reward.custom_reward_function.name=compute_score \
    +trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console"]' \
    trainer.project_name=HotpotQA_AGENT_R1 \
    trainer.experiment_name="$RUN_ID" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    trainer.log_val_generations=4 2>&1 | tee "$VALIDATION_DATA_DIR/launcher.log"

if [[ "$CONFIG_ONLY" == "1" ]]; then
    exit 0
fi

VALIDATION_DATA_DIR="$VALIDATION_DATA_DIR" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["VALIDATION_DATA_DIR"]) / "run_manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["status"] = "complete"
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("manifest marked complete")
PY
