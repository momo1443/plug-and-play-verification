# HotpotQA

The paper experiments share the certificate-based agent flow in
`recipes/hotpotqa_a9/`, with four turns, 10,000 selected training examples, and
500 updates. A1 uses outcome rewards, A9 uses the deterministic verifier, and
A6 uses a frozen LLM process judge without reference answers or annotated
supporting facts. A6 uses 50 outcome-only warmup updates, followed by a random
outcome/process reward mixture shared across each prompt group. A2 and A3 use
privileged gold-evidence supervision; A3 uses the same warmup and mixing schedule. See the
[shared training settings](../../scripts/common/README.md).

This package supplies retrieval, evidence preprocessing, and judge services.
`recipes/hotpotqa_lr/` retains shared parsing/artifact helpers and the optional
A8 experiment. The raw-answer loop supports the corresponding baseline configurations.

## Current entrypoints

```bash
bash run_pnp.sh hotpotqa --arm outcome   # outcome-only GRPO (same wrapper, A1 reward)
bash scripts/hotpotqa/pnp_hotpotqa.sh  # verifier GRPO
bash scripts/hotpotqa/run_a6.sh          # frozen process judge
bash scripts/hotpotqa/run_a2.sh          # gold process-only ablation
bash scripts/hotpotqa/run_a3.sh          # gold-evidence outcome+process ablation
bash scripts/ppo/run_terminal.sh hotpotqa
bash scripts/hotpotqa/run_a0.sh          # base-model certificate validation
```

Configure model/data/output paths and resources through the launcher's
`HOTPOTQA_*`, `CUDA_VISIBLE_DEVICES`, and `RUN_ID` environment variables.
A8 launchers are under `scripts/extensions/hotpotqa_lr/`. Run manifests record
hashes of the source files.

For standalone checkpoint evaluation, install the graders and provide an exported
model plus the frozen dataset revision:

```bash
bash scripts/evaluation/setup_official_graders.sh
HOTPOTQA_MODEL_PATH=/path/to/exported-checkpoint \
  HOTPOTQA_DATASET_REVISION=LOCAL_HOTPOTQA_SNAPSHOT \
  HOTPOTQA_EMBEDDING_MODEL=/path/to/bge-large-en-v1.5 \
  bash scripts/hotpotqa/pnp_hotpotqa_eval.sh
```

The wrapper evaluates `data/corpus/hotpotqa/validation.parquet` with the corpus
and evidence sidecar listed below. Use `HOTPOTQA_VAL_PATH`,
`HOTPOTQA_CORPUS_DATA_ROOT`, and `HOTPOTQA_EVIDENCE_SIDECAR` for alternate paths.

## Additional requirements

```bash
pip install -r recipes/hotpotqa/requirements.txt
```

Building the retrieval index requires sufficient CPU or GPU memory to encode the corpus with the configured embedding model.

## Data and resources

Expected processed files:

- `data/corpus/hotpotqa/train.parquet` (90,447 rows for formal main)
- `data/corpus/hotpotqa/validation.parquet` (7,405 rows; always retained)
- Optional cross-eval: `2wikimultihopqa_validation.parquet`, `musique_validation.parquet`
- `data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl`
- `data/corpus/hotpotqa_corpus/hpqa_corpus.npy`
- `data/corpus/hotpotqa_corpus/index.bin`
- `data/corpus/hotpotqa_corpus/hotpotqa_evidence_v1.sqlite3`

`recipes.hotpotqa.env.search_tool` queries the FAISS index. The default embedding model is `BAAI/bge-large-en-v1.5`, configured through `HOTPOTQA_EMBEDDING_MODEL`. The paper launchers use the local checkpoint at `../models/bge-large-en-v1.5`.

The paper launchers preserve the prepared paragraph, embedding, and FAISS files. The evidence sidecar is built from official distractor sentence arrays and supporting facts, aligned to the existing PID order. Evidence IDs and gold hits are audit/verifier fields and never enter the model prompt.

Paper `main` selects 10,000 training rows for 500 updates; `pilot64` and `pilot2048` select smaller prefixes. The A8 extension retains its separate 30,000-row budget.

## Data Preparation

Download the processed release from [ModelScope](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data), or regenerate:

```bash
python recipes/hotpotqa/data_preprocess/process_hotpotqa.py \
  --output_dir data/corpus/hotpotqa \
  --corpus_output_path data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl \
  --include_cross_eval

python recipes/hotpotqa/env/build_index.py \
  --data_dir data/corpus/hotpotqa_corpus \
  --corpus_path data/corpus/hotpotqa_corpus/hpqa_corpus.jsonl \
  --embedding_model /path/to/bge-large-en-v1.5
```

For local 2Wiki/MuSiQue corpus construction, use `recipes/hotpotqa/env/build_retrieval_corpus.py`.

Use the same embedding model when building the index and running retrieval.

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

Fail-closed artifact check:

```bash
python -m recipes.hotpotqa.validate_formal_a0_artifacts \
  --validation_path data/corpus/hotpotqa/validation.parquet \
  --corpus_dir data/corpus/hotpotqa_corpus \
  --evidence_sidecar_path data/corpus/hotpotqa_corpus/hotpotqa_evidence_v1.sqlite3
```

## Environment setup

Retrieval uses the local corpus and FAISS index. Configure the embedding model and device with:

```bash
export HOTPOTQA_EMBEDDING_MODEL=/path/to/bge-large-en-v1.5   # or BAAI/bge-large-en-v1.5
export HOTPOTQA_EMBEDDING_DEVICE=cpu                         # default in formal launchers
```
