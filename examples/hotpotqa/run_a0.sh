#!/usr/bin/env bash
set -euo pipefail
set -x

# Frozen A0 raw-answer validation launcher.

if (( $# != 0 )); then
    echo "This launcher takes no arguments; configure via the exported variables below" >&2
    exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/deepresearch/conda/envs/agenticrl/bin/python}"

VALIDATION_INTERFACE=raw
VALIDATION_ARM=A0
export HOTPOTQA_FORMAL_A0=1
export HOTPOTQA_FORMAL_EXPERIMENT=1
export HOTPOTQA_REWARD_ARM=A0
export HOTPOTQA_VALIDATION_INTERFACE VALIDATION_ARM

# --- Frozen validation contract knobs ----------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export HOTPOTQA_ENABLE_THINKING=false
export HOTPOTQA_FORCE_FIRST_SEARCH=false
export HOTPOTQA_REQUIRE_SENTENCE_EVIDENCE=true
export HOTPOTQA_EMBEDDING_PER_WORKER_GPU=0
export HOTPOTQA_EMBEDDING_DEVICE=cpu
export HOTPOTQA_STREAMING_RESULTS=1
export HOTPOTQA_STREAMING_FSYNC=1
export TOKENIZERS_PARALLELISM=false

export HOTPOTQA_DATA_ROOT="${HOTPOTQA_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa}"
export HOTPOTQA_CORPUS_DATA_ROOT="${HOTPOTQA_CORPUS_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa_corpus}"
export HOTPOTQA_EVIDENCE_SIDECAR="${HOTPOTQA_EVIDENCE_SIDECAR:-$HOTPOTQA_CORPUS_DATA_ROOT/hotpotqa_evidence_v1.sqlite3}"
export HOTPOTQA_EMBEDDING_MODEL="${HOTPOTQA_EMBEDDING_MODEL:-$WORKSPACE_DIR/models/bge-large-en-v1.5}"
export HOTPOTQA_MODEL_PATH="${HOTPOTQA_MODEL_PATH:-$WORKSPACE_DIR/models/Qwen3-4B}"

MODEL_PATH="$HOTPOTQA_MODEL_PATH"
TRAIN_PATH="${HOTPOTQA_TRAIN_PATH:-$HOTPOTQA_DATA_ROOT/train.parquet}"
VAL_PATH="${HOTPOTQA_VAL_PATH:-$HOTPOTQA_DATA_ROOT/validation.parquet}"
AGENT_CONFIG="$PROJECT_DIR/recipes/hotpotqa/base.yaml"
REWARD_PATH="$PROJECT_DIR/recipes/hotpotqa/reward_fn.py"

VAL_MAX_SAMPLES="${HOTPOTQA_VAL_MAX_SAMPLES:--1}"
VAL_BATCH_SIZE="${HOTPOTQA_VAL_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH_PER_STEP=1024
MAX_MODEL_LENGTH=12288
IFS=',' read -r -a CUDA_DEVICE_LIST <<<"$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${HOTPOTQA_NUM_GPUS:-${#CUDA_DEVICE_LIST[@]}}"
# Validation-only still passes through verl's train-batch divisibility check.
# Keep the historical batch of 8 when possible and round it up for GPU counts
# such as 6; no optimizer update consumes this placeholder batch.
TRAIN_BATCH_SIZE="${HOTPOTQA_TRAIN_BATCH_SIZE:-$(( (8 + NUM_GPUS - 1) / NUM_GPUS * NUM_GPUS ))}"
AGENT_WORKERS="${HOTPOTQA_AGENT_WORKERS:-$NUM_GPUS}"
VLLM_GPU_MEMORY_UTILIZATION="${HOTPOTQA_VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
VLLM_MAX_NUM_SEQS="${HOTPOTQA_VLLM_MAX_NUM_SEQS:-8}"
VAL_N="${HOTPOTQA_VAL_N:-1}"
VAL_DO_SAMPLE="${HOTPOTQA_VAL_DO_SAMPLE:-false}"
VAL_TEMPERATURE="${HOTPOTQA_VAL_TEMPERATURE:-0}"
VAL_TOP_P="${HOTPOTQA_VAL_TOP_P:-1}"
VAL_TOP_K="${HOTPOTQA_VAL_TOP_K:--1}"

if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || (( NUM_GPUS != ${#CUDA_DEVICE_LIST[@]} )); then
    echo "HOTPOTQA_NUM_GPUS must match CUDA_VISIBLE_DEVICES: num_gpus=$NUM_GPUS devices=$CUDA_VISIBLE_DEVICES" >&2
    exit 2
fi
if [[ ! "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || (( TRAIN_BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "HOTPOTQA_TRAIN_BATCH_SIZE must be positive and divisible by the GPU count: batch=$TRAIN_BATCH_SIZE num_gpus=$NUM_GPUS" >&2
    exit 2
fi
if [[ ! "$AGENT_WORKERS" =~ ^[1-9][0-9]*$ ]] || [[ ! "$VAL_N" =~ ^[1-9][0-9]*$ ]]; then
    echo "HOTPOTQA_AGENT_WORKERS and HOTPOTQA_VAL_N must be positive integers" >&2
    exit 2
fi
case "${VAL_DO_SAMPLE,,}" in
    true|false) ;;
    *)
        echo "HOTPOTQA_VAL_DO_SAMPLE must be true or false" >&2
        exit 2
        ;;
esac

RUN_ID="${RUN_ID:-qwen3-4b_${VALIDATION_ARM,,}_${VALIDATION_INTERFACE}_full7k_${NUM_GPUS}gpu_$(date +%Y%m%d-%H%M%S)}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$WORKSPACE_DIR/logs/$RUN_ID}"
# Keep Ray's tmp path short: AF_UNIX socket paths must stay <=107 bytes, and the
# long RUN_ID overflows the plasma_store socket path if used as the Ray tmpdir.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_a0}"

mkdir -p "$VALIDATION_DATA_DIR"
cd "$PROJECT_DIR"

# --- Write a reproducibility manifest before launch ---------------------------
RUN_ID="$RUN_ID" \
VALIDATION_DATA_DIR="$VALIDATION_DATA_DIR" \
MODEL_PATH="$MODEL_PATH" \
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
VAL_N="$VAL_N" \
VAL_DO_SAMPLE="$VAL_DO_SAMPLE" \
VAL_TEMPERATURE="$VAL_TEMPERATURE" \
VAL_TOP_P="$VAL_TOP_P" \
VAL_TOP_K="$VAL_TOP_K" \
NUM_GPUS="$NUM_GPUS" \
AGENT_WORKERS="$AGENT_WORKERS" \
PROJECT_DIR="$PROJECT_DIR" \
VALIDATION_ARM="$VALIDATION_ARM" \
VALIDATION_INTERFACE="$VALIDATION_INTERFACE" \
HOTPOTQA_CORPUS_DATA_ROOT="$HOTPOTQA_CORPUS_DATA_ROOT" \
HOTPOTQA_EVIDENCE_SIDECAR="$HOTPOTQA_EVIDENCE_SIDECAR" \
HOTPOTQA_EMBEDDING_MODEL="$HOTPOTQA_EMBEDDING_MODEL" \
"$PYTHON_BIN" - <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from recipes.hotpotqa.evidence import EVIDENCE_SCHEMA_VERSION
from recipes.hotpotqa.final_answer_protocol import resolve_final_answer_protocol
from recipes.hotpotqa.prepare_formal_rlvr_run import (
    _model_identity,
    _package_versions,
)

validation_arm = os.environ["VALIDATION_ARM"]
validation_interface = os.environ["VALIDATION_INTERFACE"]
if validation_arm != "A0" or validation_interface != "raw":
    raise ValueError("Only the frozen A0 raw validation interface is supported")
final_answer_protocol = resolve_final_answer_protocol()
project_dir = Path(os.environ["PROJECT_DIR"])
out_dir = Path(os.environ["VALIDATION_DATA_DIR"])
corpus_dir = Path(os.environ["HOTPOTQA_CORPUS_DATA_ROOT"])
train_path = Path(os.environ["TRAIN_PATH"])
val_path = Path(os.environ["VAL_PATH"])
model_dir = Path(os.environ["MODEL_PATH"])
embedding_model = Path(os.environ.get("HOTPOTQA_EMBEDDING_MODEL", ""))
evidence_sidecar = Path(os.environ["HOTPOTQA_EVIDENCE_SIDECAR"])


def sha256(path: Path, chunk=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def light(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "present": False}
    return {"path": str(p.resolve()), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}


def full(path: Path) -> dict:
    p = Path(path)
    rec = light(p)
    if rec.get("present", True) is not False:
        rec["sha256"] = sha256(p)
    return rec


def parquet_dataset(path: Path, *, role: str) -> dict:
    rec = full(path)
    if rec.get("present") is False:
        rec["role"] = role
        rec["num_rows"] = None
        return rec
    rec["role"] = role
    rec["num_rows"] = int(pq.read_metadata(path).num_rows)
    rec["dataset"] = "hotpotqa"
    return rec


train_rows = int(pq.read_metadata(train_path).num_rows) if train_path.is_file() else None
val_rows = int(pq.read_metadata(val_path).num_rows) if val_path.is_file() else None
val_max_samples = int(os.environ["VAL_MAX_SAMPLES"])
if val_rows is None:
    selected_val_rows = None
elif val_max_samples < 0:
    selected_val_rows = val_rows
else:
    selected_val_rows = min(val_max_samples, val_rows)

code_rel = [
    "recipes/hotpotqa/hotpotqa_agent_flow.py",
    "recipes/hotpotqa/final_answer_protocol.py",
    "recipes/hotpotqa/reward_fn.py",
    "recipes/hotpotqa/prompts.py",
    "recipes/hotpotqa/base.yaml",
    "recipes/hotpotqa/evidence.py",
    "recipes/hotpotqa/env/search_tool.py",
    "agent_r1/trainer/streaming_agent_validation.py",
    "agent_r1/trainer/compact_validation_record.py",
    "examples/hotpotqa/run_a0.sh",
]

cuda_devices = [
    value.strip()
    for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if value.strip()
]

manifest = {
    "contract_version": "hotpotqa-formal-a0-raw-final-v3",
    "run_id": os.environ["RUN_ID"],
    "status": "prepared",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "arm": validation_arm,
    "run_mode": "validation_only",
    "validation_interface": validation_interface,
    "final_answer_protocol": final_answer_protocol,
    "output_dir": str(out_dir.resolve()),
    "output_jsonl": str((out_dir / "0.jsonl").resolve()),
    "resources": {
        "num_gpus": int(os.environ["NUM_GPUS"]),
        "agent_workers": int(os.environ["AGENT_WORKERS"]),
        "cuda_visible_devices": cuda_devices,
        "vllm_gpu_memory_utilization": float(os.environ["VLLM_GPU_MEMORY_UTILIZATION"]),
        "vllm_max_num_seqs": int(os.environ["VLLM_MAX_NUM_SEQS"]),
        "val_batch_size": int(os.environ["VAL_BATCH_SIZE"]),
    },
    "scientific_config": {
        "validation_only": True,
        "optimizer_updates": 0,
        "thinking_mode": "disabled",
        "final_answer_protocol": final_answer_protocol,
        "force_first_search": False,
        "max_steps": 4,
        "max_model_searches": 3,
        "max_parallel_calls": 1,
        "tool_parser": "hermes",
        "retrieval_top_k": 5,
        "max_prompt_tokens_per_step": int(os.environ["MAX_PROMPT_LENGTH"]),
        "max_response_tokens_per_step": int(os.environ["MAX_RESPONSE_LENGTH_PER_STEP"]),
        "engine_max_model_len": int(os.environ["MAX_MODEL_LENGTH"]),
        "validation_n": int(os.environ["VAL_N"]),
        "validation_do_sample": os.environ["VAL_DO_SAMPLE"].lower() == "true",
        "validation_temperature": float(os.environ["VAL_TEMPERATURE"]),
        "validation_top_p": float(os.environ["VAL_TOP_P"]),
        "validation_top_k": int(os.environ["VAL_TOP_K"]),
        "terminal_reward": "normalized exact match (evaluation only)",
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
    },
    "model": _model_identity(model_dir),
    "embedding_model": {
        "path": str(embedding_model.resolve()) if embedding_model else "",
        "name": embedding_model.name if embedding_model else "",
        "device": os.environ.get("HOTPOTQA_EMBEDDING_DEVICE", "cpu"),
        "per_worker_gpu": os.environ.get("HOTPOTQA_EMBEDDING_PER_WORKER_GPU", "0"),
    },
    "data": {
        "dataset": "hotpotqa",
        "train_parquet": parquet_dataset(train_path, role="train_source"),
        "validation_parquet": parquet_dataset(val_path, role="validation_eval"),
        "source_train_rows": train_rows,
        "source_validation_rows": val_rows,
        "val_max_samples_requested": val_max_samples,
        "validation_rows_selected": selected_val_rows,
        "note": (
            f"{validation_arm} interface validation only; train.parquet is recorded for "
            "contract parity but not optimized on."
        ),
    },
    "selection": {
        "validation": {
            "split": "validation",
            "count": selected_val_rows,
            "requested_max_samples": val_max_samples,
            "source_rows": val_rows,
        }
    },
    "artifacts": {
        "train_parquet": full(train_path) if train_path.is_file() else light(train_path),
        "validation_parquet": full(val_path),
        "corpus_jsonl": light(corpus_dir / "hpqa_corpus.jsonl"),
        "corpus_embeddings": light(corpus_dir / "hpqa_corpus.npy"),
        "faiss_index": light(corpus_dir / "index.bin"),
        "evidence_sidecar": light(evidence_sidecar),
        "embedding_model": str(embedding_model.resolve()) if embedding_model else "",
    },
    "code": {rel: full(project_dir / rel) for rel in code_rel},
    "package_versions": _package_versions(),
}
tmp = out_dir / ".run_manifest.json.tmp"
tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, out_dir / "run_manifest.json")
print("wrote manifest ->", out_dir / "run_manifest.json")
print(
    f"model={manifest['model']['name']} "
    f"val_rows={selected_val_rows}/{val_rows} "
    f"train_rows={train_rows}"
)
PY

# --- Validation-only pass (no optimizer update) -------------------------------
"$PYTHON_BIN" -m agent_r1.trainer.main_agent_ppo \
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
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
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
    actor_rollout_ref.rollout.agent.default_agent_flow=hotpotqa_agent \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE" \
    actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
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

# --- Mark manifest complete ---------------------------------------------------
VALIDATION_DATA_DIR="$VALIDATION_DATA_DIR" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["VALIDATION_DATA_DIR"]) / "run_manifest.json"
m = json.loads(p.read_text(encoding="utf-8"))
m["status"] = "complete"
p.write_text(json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("manifest marked complete")
PY
