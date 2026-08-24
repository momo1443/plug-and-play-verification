# HotpotQA Recipe

## Overview

This recipe runs the retained raw-answer HotpotQA arms with local FAISS/BGE retrieval. The standalone local-reasoning experiment lives under `recipes/hotpotqa_lr/`, and the certificate-grounded A9 experiment lives under `recipes/hotpotqa_a9/`.

| Arm | Role | Optimizer signal |
|-----|------|------------------|
| **A0** | Evaluation-only baseline | No policy update; process rewards are audit-only |
| **A1** | Terminal-only RLVR | GRPO + `step_causal`; reward = terminal exact match |
| **A2** | Verifier-only agentic RLVR | GRPO + `step_causal`; per-step new-gold-evidence reward; final-answer tokens masked from policy loss |
| **A3** | Combined agentic RLVR | GRPO + `step_causal`; `0.5 * process + 0.5 * terminal EM`; final-answer tokens remain in policy loss |
| **A6** | Gold-conditioned semantic verifier | frozen LLM fact-coverage Judge + terminal EM |
| **A7** | Weak execution control | `1/3` per valid observable search + terminal EM |
| **A9** | Certificate-grounded agentic RLVR | Minimal machine-checkable certificates per action; three-arm experiment under `recipes/hotpotqa_a9/` |

Official dataset references: https://hotpotqa.github.io/ and https://github.com/StonyBrookNLP/musique. Processed Agent-R1 assets are also available from the [Agent-R1-data ModelScope release](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data).

## Directory Layout

Recipe code (`recipes/hotpotqa/`):

- `base.yaml` — agent config (`max_steps=4`, retrieval paths, evidence sidecar)
- `hotpotqa_agent_flow.py` — shared rollout loop for every formal arm
- `reward_arm.py` — frozen reward and final-token mask semantics
- `reward_fn.py` — terminal exact-match scorer
- `process_verifier.py` — deterministic evidence process reward
- `a9_behavioral_verifier.py` / `a9_entity_library.json` — **DEPRECATED** A9 counterfactual probe generator (retained for artifact replay; current A9 lives in `recipes/hotpotqa_a9/`)
- `judge_server.py` — shared frozen-Judge client (local vLLM or remote API) with retries and exact-input cache
- `final_answer_protocol.py` — raw-final contract for retained arms
- `prepare_formal_rlvr_run.py` / `validate_formal_a0_artifacts.py` — fail-closed preflight + manifests
- `build_evidence_sidecar.py` — official-sentence evidence SQLite builder
- `data_preprocess/` / `env/` — parquet conversion, corpus/index build, search tool

Launch scripts (`examples/hotpotqa/`):

- `run_a0.sh` — formal A0 validation-only launcher
- `run_a1.sh`, `run_a2.sh`, `run_a3.sh`, `run_a6.sh`, `run_a7.sh`, `run_a9.sh` — retained arm entrypoints
- `run_rlvr.sh` — shared GRPO training engine
- `run_with_judge.sh` — A6 Judge lifecycle wrapper

## Additional Requirements

```bash
pip install -r recipes/hotpotqa/requirements.txt
```

Building the retrieval index needs enough CPU/GPU memory to encode the corpus with the configured embedding model.

## Data And Resources

Expected processed files:

- `data/corpus/hotpotqa/train.parquet` (90,447 rows for formal main)
- `data/corpus/hotpotqa/validation.parquet` (7,405 rows; always retained)
- Optional cross-eval: `2wikimultihopqa_validation.parquet`, `musique_validation.parquet`
- `data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl`
- `data/corpus/hotpotqa_corpus/hpqa_corpus.npy`
- `data/corpus/hotpotqa_corpus/index.bin`
- `data/corpus/hotpotqa_corpus/hotpotqa_evidence_v1.sqlite3`

The FAISS index is searched by `recipes.hotpotqa.env.search_tool`. Default embedding id is `BAAI/bge-large-en-v1.5` (`HOTPOTQA_EMBEDDING_MODEL`); formal launchers point at the local checkpoint under `../models/bge-large-en-v1.5`.

Formal A0/A1/A2/A3 preserve paragraph, embedding, and FAISS files byte-for-byte. The evidence sidecar is built from official distractor sentence arrays and supporting facts, aligned to the existing PID order. Evidence IDs and gold hits are audit/verifier fields and never enter the model prompt.

Formal `pilot64`, `pilot2048`, and `main` modes use deterministic prefixes of 64, 2,048, and 30,000 training rows respectively, and retain all 7,405 validation rows.

## Data Preparation

Download the processed release from [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data), or regenerate:

```bash
python recipes/hotpotqa/data_preprocess/process_hotpotqa.py \
  --output_dir data/corpus/hotpotqa \
  --corpus_output_path data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl \
  --include_cross_eval

python recipes/hotpotqa/env/build_index.py \
  --data_dir data/corpus/hotpotqa_corpus \
  --corpus_path data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl
```

For local 2Wiki/MuSiQue corpus construction, use `recipes/hotpotqa/env/build_retrieval_corpus.py`.

Build the verifier sidecar from official Hugging Face Parquet shards (e.g. under `../.cache/hotpotqa-official`):

```bash
python -m recipes.hotpotqa.build_evidence_sidecar \
  --train_source ../.cache/hotpotqa-official/train \
  --validation_source ../.cache/hotpotqa-official/validation \
  --train_parquet data/corpus/hotpotqa/train.parquet \
  --validation_parquet data/corpus/hotpotqa/validation.parquet \
  --corpus_path data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl \
  --output_path data/corpus/hotpotqa_corpus/hotpotqa_evidence_v1.sqlite3
```

Fail-closed artifact check (also invoked by formal launchers):

```bash
python -m recipes.hotpotqa.validate_formal_a0_artifacts \
  --validation_path data/corpus/hotpotqa/validation.parquet \
  --corpus_dir data/corpus/hotpotqa_corpus \
  --evidence_sidecar_path data/corpus/hotpotqa_corpus/hotpotqa_evidence_v1.sqlite3
```

## Environment Setup

No separate HTTP service. Retrieval loads local corpus + FAISS. Typical knobs:

```bash
export HOTPOTQA_EMBEDDING_MODEL=/path/to/bge-large-en-v1.5   # or BAAI/bge-large-en-v1.5
export HOTPOTQA_EMBEDDING_DEVICE=cpu                         # default in formal launchers
```

## Formal A0 (validation only)

No policy update. Shared `HotpotQAAgentFlow`: thinking off, Hermes parsing, top-5 retrieval, one search per turn, three search turns + one final turn, `force_first_search=false`, raw final completion. Process rewards are recorded as replayable audit fields only.

Defaults in `run_a0.sh`: 4 GPUs (`4,5,6,7`), model `../models/Qwen3-4B`.

```bash
# Full 7,405-row validation
bash examples/hotpotqa/run_a0.sh

```

Override model/GPUs via env, e.g. `HOTPOTQA_MODEL_PATH=../models/Qwen3.5-4B CUDA_VISIBLE_DEVICES=0,1,2,3`.

Every A0 JSONL row carries stable `sample_key`, qid, search steps, evidence IDs, process-reward audit, final answer, and terminal EM. The launcher writes a run manifest before allocating the model runtime.

## Formal A1 / A2 / A3 (GRPO training)

All three training arms use `run_rlvr.sh` with GRPO, `algorithm.grpo.credit_assignment=step_causal`, and `gamma=1.0`. They uniformly enable verl/GRPO's built-in actor-loss reference KL (`low_var_kl`, coefficient `0.001`) while keeping KL out of the task reward. Entrypoints:

- `run_a1.sh` → `HOTPOTQA_REWARD_ARM=A1` (wrapper default GPUs `0,1,2,3,4,5,6,7`)
- `run_a2.sh` → `HOTPOTQA_REWARD_ARM=A2` (wrapper default GPUs `0,1,2,3,4,5,6,7`)
- `run_a3.sh` → `HOTPOTQA_REWARD_ARM=A3` (combined `0.5/0.5`, final-answer mask = 1; GPU selection is explicit per run)

The launcher accepts `pilot64`, `pilot2048`, and `main`. Defaults are model `../models/Qwen3.5-4B`, rollout `n=4`, and one pass over the selected deterministic prefix. Performance-sensitive defaults enable actor dynamic batching and keep actor parameter/optimizer offload disabled. Calling `run_rlvr.sh` directly requires `HOTPOTQA_REWARD_ARM`.

Main training shape (overridable via `HOTPOTQA_*`): `train_batch_size=20`, `rollout_n=4`, `grpo_micro_batch_size_per_gpu=2`, actor token cap 8,192, entropy objective disabled, first 30,000 train / all 7,405 validation rows. Use `HOTPOTQA_GRPO_MICRO_BATCH_SIZE` to change the micro-batch size.

```bash
# Main runs (default: 1,500 steps)
bash examples/hotpotqa/run_a1.sh
HOTPOTQA_TOTAL_TRAINING_STEPS=800 bash examples/hotpotqa/run_a2.sh
bash examples/hotpotqa/run_a3.sh

# Example: Qwen3.5-4B on GPUs 2–7
HOTPOTQA_MODEL_PATH=../models/Qwen3.5-4B \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
  bash examples/hotpotqa/run_a3.sh
```

Formal A1/A2/A3 entrypoints reject trailing Hydra overrides. Configure via `HOTPOTQA_*` / `RUN_ID` / `CUDA_VISIBLE_DEVICES` so the preflight manifest records arm, reward weights, final response mask, splits, estimator, model hashes, artifacts, and resources. Outputs land under `../logs/$RUN_ID/` (`train.log`, `tensorboard/`, `checkpoints/`, one append-only `rollouts.jsonl`, `validation/`, and `run_manifest.json`).

Useful env flags: `HOTPOTQA_SKIP_PREFLIGHT=1` reuses a complete manifest for the same experiment directory; `HOTPOTQA_HYDRA_CONFIG_ONLY=1` only expands configuration and never starts training.

Formal A1/A2/A3 training saves actor checkpoints every 100 optimizer steps and retains the latest two by default. Override with `HOTPOTQA_SAVE_FREQ` and `HOTPOTQA_MAX_ACTOR_CKPT_TO_KEEP` when a different cadence is required; values are captured at launch and do not alter an already-running process.

## Legacy Algorithm Scripts

Upstream-style launchers (accept Hydra `"$@"` overrides; not the formal A0/A1/A2/A3 contract):

```bash
bash examples/hotpotqa/run_ppo.sh
bash examples/hotpotqa/run_steppo.sh
bash examples/hotpotqa/run_grpo.sh
bash examples/hotpotqa/run_rloo.sh
bash examples/hotpotqa/run_reinforce.sh
bash examples/hotpotqa/run_gspo.sh
bash examples/hotpotqa/run_gigpo.sh
```

## Core Code Entry Points

- Shared rollout: `hotpotqa_agent_flow.py`
- Arm semantics: `reward_arm.py`
- Terminal reward: `reward_fn.py`
- Process verifier: `process_verifier.py`
- Final-answer contract: `final_answer_protocol.py`
- Retrieval: `env/search_tool.py`
- Prompts / tool schema: `prompts.py`

## Outputs And Evaluation

Validation scores normalized exact match against `reward_model.ground_truth`. Search behavior is controlled by `max_steps`, `max_parallel_calls`, and `force_first_search` in `base.yaml` (formal runs freeze thinking off and `force_first_search=false`).

## References

- HotpotQA: https://hotpotqa.github.io/
- MuSiQue: https://github.com/StonyBrookNLP/musique
