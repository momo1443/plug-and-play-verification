# HotpotQA Recipe

## Overview

This recipe trains a multi-hop question-answering agent with a retrieval tool. The agent searches a FAISS index built from passage text and returns the final answer in the format expected by the HotpotQA reward function.

Official dataset references: https://hotpotqa.github.io/ and https://github.com/StonyBrookNLP/musique. Processed Agent-R1 train, validation, cross-eval, and retrieval assets for this recipe are available from the [Agent-R1-data ModelScope release](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data).

## Directory Layout

- `base.yaml`: HotpotQA agent configuration.
- `hotpotqa_agent_flow.py`: Agent-R1 rollout loop for retrieval-based QA.
- `data_preprocess/process_hotpotqa.py`: Converts HotpotQA and optional cross-eval splits into Agent-R1 parquet files.
- `env/build_retrieval_corpus.py`: Builds retrieval corpora from local 2Wiki/MuSiQue-style raw files.
- `env/build_index.py`: Encodes the HotpotQA corpus and builds the FAISS index.
- `env/search_tool.py`: Runtime FAISS/BGE retrieval tool.
- `examples/hotpotqa/*.sh`: Training launch scripts for PPO, StepPO, GRPO, RLOO, REINFORCE, GSPO, and GiGPO variants.

## Additional Requirements

Install recipe-specific extras after setting up the base Agent-R1 / verl environment:

```bash
pip install -r recipes/hotpotqa/requirements.txt
```

Building the retrieval index requires enough CPU/GPU memory to encode the corpus with the configured embedding model.

## Data And Resources

Expected processed files:

- `data/corpus/hotpotqa/train.parquet`
- `data/corpus/hotpotqa/validation.parquet`
- Optional cross-eval parquets such as `data/corpus/hotpotqa/2wikimultihopqa_validation.parquet` and `data/corpus/hotpotqa/musique_validation.parquet`
- `data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl`
- `data/corpus/hotpotqa_corpus/hpqa_corpus.npy`
- `data/corpus/hotpotqa_corpus/index.bin`

The FAISS index is searched by `recipes.hotpotqa.env.search_tool`. The default embedding model is `BAAI/bge-large-en-v1.5`, configurable with `HOTPOTQA_EMBEDDING_MODEL`.

Formal A0 preserves those paragraph, embedding, and FAISS files byte-for-byte. It additionally requires `hotpotqa_evidence_v1.sqlite3`, built from the official distractor sentence arrays and supporting facts and exactly aligned to the existing PID order. No embedding or index rebuild occurs. Evidence IDs and gold hits remain audit-only and never enter the model prompt.

Formal A1/A2 use the current Agent-R1 HotpotQA split contract: all 90,447 training rows are the main-run source and all 7,405 validation rows are retained for evaluation. A smoke may cap the number of training questions but never substitutes a validation subset.

## Data Preparation

Download the processed release from [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data), then place or symlink the HotpotQA files to the paths above. To regenerate HotpotQA-style files from public sources for local testing:

```bash
python recipes/hotpotqa/data_preprocess/process_hotpotqa.py \
  --output_dir data/corpus/hotpotqa \
  --corpus_output_path data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl \
  --include_cross_eval

python recipes/hotpotqa/env/build_index.py \
  --data_dir data/corpus/hotpotqa_corpus \
  --corpus_path data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl
```

For local 2Wiki/MuSiQue corpus construction, use `recipes/hotpotqa/env/build_retrieval_corpus.py` with raw files placed under the paths expected by that script.

Build the versioned verifier sidecar from the official Hugging Face Parquet shards after downloading them under a project-local cache:

```bash
python -m recipes.hotpotqa.build_evidence_sidecar \
  --train_source ../.cache/hotpotqa-official/train \
  --validation_source ../.cache/hotpotqa-official/validation \
  --train_parquet data/corpus/hotpotqa/train.parquet \
  --validation_parquet data/corpus/hotpotqa/validation.parquet \
  --corpus_path data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl \
  --output_path data/corpus/hotpotqa_corpus/hotpotqa_evidence_v1.sqlite3
```

The formal launcher performs a fail-closed count, checksum, mapping-rate, sidecar, corpus, embedding, and index check. It can also be run directly:

```bash
python -m recipes.hotpotqa.validate_formal_a0_artifacts \
  --validation_path data/corpus/hotpotqa/validation.parquet \
  --corpus_dir data/corpus/hotpotqa_corpus \
  --evidence_sidecar_path data/corpus/hotpotqa_corpus/hotpotqa_evidence_v1.sqlite3
```

## Formal A0

Formal A0 is evaluation-only: no policy update or verifier reward is applied. A0 and future A2 use the same `HotpotQAAgentFlow`; the old provisional native-AgentLoop A0 path has been removed. The flow freezes thinking off, Hermes parsing, top-5 paragraph retrieval, one search per turn, three search turns plus one final turn, and `force_first_search=false`. A0 records deterministic process rewards only as replayable audit fields.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/hotpotqa/run_a0.sh
```

For a four-question gate before the full run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HOTPOTQA_VAL_MAX_SAMPLES=4 \
RUN_ID=qwen35-4b_a0_smoke \
  bash examples/hotpotqa/run_a0.sh
```

Every A0 JSONL row contains stable `sample_key`, official qid, ordered native-thinking audit, model-generated search steps, exact source sentence records, gold/new/covered IDs, offline process rewards, unresolved source annotations, evidence metrics, final answer, and terminal EM. The launcher freezes an artifact lock and ordered-qid run manifest before allocating the model runtime; completion requires strict sidecar replay and exact row/qid completeness.

## Environment Setup

No separate HTTP service is required. The retrieval tool loads local corpus and FAISS files. Typical environment variables:

```bash
export HOTPOTQA_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

By default, retrieval uses `HOTPOTQA_EMBEDDING_DEVICE=cpu` unless configured otherwise.

## Training Scripts

Formal Qwen3.5 A1/A2 use GRPO only. A1 has zero search rewards and terminal exact-match; A2 uses deterministic new-gold-evidence rewards, zero final reward, and masks final-answer tokens from its process-policy loss. Both use step-causal group-relative advantages with `gamma=1.0`.

```bash
# CPU/artifact preflight only
CUDA_VISIBLE_DEVICES=4,5,6,7 \
HOTPOTQA_PREFLIGHT_ONLY=1 \
HOTPOTQA_RUN_MODE=smoke \
  bash examples/hotpotqa/run_a1.sh

# Four-GPU, one-update compatibility smoke
CUDA_VISIBLE_DEVICES=4,5,6,7 HOTPOTQA_RUN_MODE=smoke \
  bash examples/hotpotqa/run_a1.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 HOTPOTQA_RUN_MODE=smoke \
  bash examples/hotpotqa/run_a2.sh
```

Main runs are fail-closed to the full 90,447/7,405 source contract and require an explicit optimizer-step budget:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
HOTPOTQA_RUN_MODE=main \
HOTPOTQA_TOTAL_TRAINING_STEPS=<frozen_step_budget> \
  bash examples/hotpotqa/run_a1.sh
```

The formal entrypoints reject trailing Hydra overrides. Use `HOTPOTQA_*` environment variables so the manifest can record the resolved arm, data counts, estimator, model, artifacts, code hashes, and resource mapping.

```bash
bash examples/hotpotqa/run_ppo.sh
bash examples/hotpotqa/run_steppo.sh
bash examples/hotpotqa/run_grpo.sh
bash examples/hotpotqa/run_rloo.sh
bash examples/hotpotqa/run_reinforce.sh
bash examples/hotpotqa/run_gspo.sh
bash examples/hotpotqa/run_gigpo.sh
```

Scripts accept trailing Hydra overrides through `"$@"`.

## Core Code Entry Points

- Shared A0/A2 rollout flow: `recipes/hotpotqa/hotpotqa_agent_flow.py`.
- Retrieval utilities: `recipes/hotpotqa/env/search_tool.py`.
- Prompt and tool schema: `recipes/hotpotqa/prompts.py`.
- Reward: `recipes/hotpotqa/reward_fn.py`.

## Outputs And Evaluation

Validation uses normalized exact-match style reward against `reward_model.ground_truth`. Search behavior is controlled by `max_steps`, `max_parallel_calls`, and `force_first_search` in the recipe config.

## References

- HotpotQA: https://hotpotqa.github.io/
- MuSiQue: https://github.com/StonyBrookNLP/musique
