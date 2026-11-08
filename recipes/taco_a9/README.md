# TACO training data

The paper launchers use the first 10,000 rows of the original `ALL` training
split, in shard-name and row order. Format conversion preserves repeated
questions and performs no filtering, shuffling, or batch-size trimming.
The next 500 rows provide local diagnostics without changing the training prefix.
LiveCodeBench remains the final evaluation benchmark.

Place the original training shards under `data/raw/TACO/ALL/`, then run from
the repository root:

```bash
python -m recipes.taco_a9.prepare_data \
  --taco-dir data/raw/TACO \
  --output-dir ../data/coding_benchmarks/taco_a1_prepared
python -m recipes.taco_a9.build_sidecar_index \
  --sidecar ../data/coding_benchmarks/taco_a1_prepared/taco_a9_sidecar.jsonl \
  --output ../data/coding_benchmarks/taco_a1_prepared/taco_a9_sidecar.index.json
```

The default selection is `--train-samples 10000`. Converted Parquet files retain
source row indices and selection metadata. Both outcome-only and verifier
training validate these records before starting. Regenerate files prepared with
an earlier conversion policy using the commands above.

The runner-only sidecar contains test cases and function names, without reference
solutions. Standard-input programs and call-based functions or `Solution` methods
run inside the same bubblewrap sandbox. Function names are also included in the
task prompt to specify the required interface; test answers stay in the sidecar.
Call-based arguments and expected outputs retain their JSON values. The function
result comparison supports TACO's
[direct and wrapped-output conventions](https://github.com/FlagOpen/TACO/blob/main/metrics/testing_util.py).

Developer and hidden tests are disjoint. Tasks with at least four tests retain
at least two tests in each suite; two- and three-test tasks use one developer
test. A single test is reserved for hidden grading. Tasks with no tests remain
in the source prefix and receive zero test-success reward. Missing required
fields or malformed test records raise an error without skipping or replacing
the source row.
