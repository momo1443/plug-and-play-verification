"""Select an exact training prefix after checking rendered prompt lengths."""

import argparse
import hashlib
import json
from pathlib import Path


def select_prompt_indices(rows, tokenizer, sample_count, max_prompt_length):
    if sample_count < 1 or max_prompt_length < 1:
        raise ValueError("Sample count and prompt limit must be positive")
    if len(rows) < sample_count:
        raise ValueError(f"Only {len(rows)} source rows; need {sample_count}")
    for index, row in enumerate(rows[:sample_count]):
        token_ids = tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=True, tokenize=True, return_dict=False
        )
        if len(token_ids) > max_prompt_length:
            raise ValueError(
                f"Training prefix row {index} has {len(token_ids)} prompt tokens; "
                f"limit is {max_prompt_length}. Source-prefix selection cannot skip or replace rows."
            )
    return list(range(sample_count)), []


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    args = parser.parse_args()
    if args.samples < 1 or args.max_prompt_length < 1:
        parser.error("samples and max-prompt-length must be positive")
    table = pq.read_table(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    selected, excluded = select_prompt_indices(
        table.select(["prompt"]).to_pylist(), tokenizer, args.samples, args.max_prompt_length
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(pa.array(selected)), args.output)
    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "model": str(args.model.resolve()),
        "samples": len(selected),
        "max_prompt_length": args.max_prompt_length,
        "selection": "exact source prefix; row order and labels unchanged; overlong rows are errors",
        "selected_rows": selected,
        "excluded_overlong_rows": excluded,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(selected)} training rows; skipped {len(excluded)} overlong prompts: {args.output}")


if __name__ == "__main__":
    main()
