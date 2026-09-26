# HotpotQA

The paper arms share the certificate flow in `recipes/hotpotqa_a9/`: four turns,
10,000 selected training rows, and 500 updates. A1 uses outcome reward; A9 uses
the deterministic verifier; A6 uses a frozen causal process judge without gold
answers. A2/A3 are explicitly privileged gold-evidence ablations. See the
[paper contract](../../docs/paper-contract.md).

This package supplies retrieval, evidence preprocessing, and judge services.
`recipes/hotpotqa_lr/` retains shared parsing/artifact helpers and the optional
A8 experiment. The earlier raw-answer loop is retained for legacy configurations;
its [historical guide](../../docs/archive/hotpotqa-legacy.md) is archived.

## Current entrypoints

```bash
bash scripts/hotpotqa/run_a1.sh          # outcome-only GRPO
bash scripts/hotpotqa/run_a9_uniform.sh  # verifier GRPO
bash scripts/hotpotqa/run_a6.sh          # frozen process judge
bash scripts/hotpotqa/run_a2.sh          # gold process-only ablation
bash scripts/hotpotqa/run_a3.sh          # gold 0.5 mixture ablation
bash scripts/ppo/run_terminal.sh hotpotqa
bash scripts/hotpotqa/run_a0.sh          # base-model certificate validation
bash scripts/hotpotqa/run_validation.sh  # certificate checkpoint validation
```

Configure model/data/output paths and resources through the launcher's
`HOTPOTQA_*`, `CUDA_VISIBLE_DEVICES`, and `RUN_ID` environment variables.
A8 launchers are under `scripts/extensions/hotpotqa_lr/`. Source hashing in
run manifests uses the current script paths.

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
