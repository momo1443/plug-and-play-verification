"""Audit every source row and quarantine conflicting or incomplete problems.

Original JSON/parquet files stay intact. Cleaned splits retain source indices,
source order and the original held-out membership; labels are not inferred
from policy predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from recipes.deepscaler.math_expressions import answers_equal, canonical_answer

DRESS_PROBLEM = (r"A dress originally priced at $80$ dollars was put on sale for $25\%$ off.  "
                 r"If $10\%$ tax was added to the sale price, then the total selling price (in dollars) of the dress was")
CORRECTIONS = {DRESS_PROBLEM: {"answer": "66", "proof": "80 * (1 - 25/100) * (1 + 10/100) = 66"}}


def answer_parse_status(answer: str) -> tuple[str, str]:
    from recipes.deepscaler.math_expressions import _parsed_answer, evaluation, latex_to_python, numeric_tree
    answer = canonical_answer(answer)
    converted = latex_to_python(answer)
    if converted and numeric_tree(converted):
        value, status = evaluation(converted)
        if status != 'limit': return answer, "numeric" if value is not None else "invalid_numeric_domain"
    if "..." in answer or "…" in answer:
        return answer, "ambiguous_ellipsis"
    try:
        return answer, "symbolic_or_structured" if _parsed_answer(answer) else "unsupported_answer_format"
    except (ValueError, TypeError, TimeoutError, RecursionError):
        return answer, "unsupported_answer_format"


def quality_decisions(records: list[dict]) -> dict:
    groups = defaultdict(list)
    excluded, corrected, conflicts, equivalent_groups = {}, {}, [], []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            excluded[index] = "invalid_record"; continue
        q, a = record.get("problem"), record.get("answer")
        if not isinstance(q, str) or not q.strip() or not isinstance(a, (str, int, float)) or str(a).strip().lower() in {"", "none", "null", "nan", "inf", "..."}:
            excluded[index] = "missing_or_invalid_problem_or_answer"; continue
        groups[q.strip()].append(index)
        # A figure reference without a supplied diagram has no reproducible
        # text-only solution. Asymptote descriptions are retained as evidence.
        if (re.search(r"\b(?:diagram|figure|polygon)\s+(?:shown|below)|\b(?:shown|pictured)\s+(?:below|in the figure)|\bin the (?:diagram|figure)\b", q, re.I)
                and not re.search(r"\[asy\]|<svg|data:image", q, re.I)):
            excluded[index] = "missing_referenced_diagram"
        if q.strip() in CORRECTIONS:
            corrected[index] = CORRECTIONS[q.strip()]
    for q, indices in groups.items():
        labels = list(dict.fromkeys(canonical_answer(str(records[i]["answer"])) for i in indices))
        if len(labels) < 2:
            continue
        if all(answers_equal(a, labels[0]) for a in labels[1:]):
            equivalent_groups.append({"indices": indices, "labels": labels})
        else:
            conflicts.append({"indices": indices, "problem": q, "labels": labels})
            for i in indices: excluded[i] = "conflicting_reference_answers"
    return {"source_rows": len(records), "unique_problems": len(groups),
            "excluded": excluded, "corrections": corrected,
            "conflicting_groups": conflicts, "equivalent_label_groups": equivalent_groups,
            "limitations": "Every row is structurally checked and every duplicate-label conflict compared. This is not a proof of every single-label reference answer."}


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, default=Path("data/raw/DeepScaleR-Preview-Dataset/deepscaler.json"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/corpus/deepscaler"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.raw_json.read_text())
    report = quality_decisions(records)
    # Every distinct label is checked, including labels outside the rollout
    # prefix. CPU processes isolate parser timeouts; no model is involved.
    import multiprocessing
    labels = sorted({canonical_answer(str(r.get('answer'))) for r in records if isinstance(r, dict) and r.get('answer') is not None})
    statuses = {}
    with multiprocessing.get_context('spawn').Pool(4) as pool:
        for number, (label, status) in enumerate(pool.imap_unordered(answer_parse_status, labels, chunksize=16), 1):
            statuses[label] = status
            if number % 1000 == 0: print(f"Parsed {number}/{len(labels)} distinct reference answers", flush=True)
    for index, record in enumerate(records):
        if not isinstance(record, dict) or index in report['excluded'] or index in report['corrections']: continue
        status = statuses.get(canonical_answer(str(record.get('answer'))))
        if status in {'invalid_numeric_domain', 'ambiguous_ellipsis', 'unsupported_answer_format'}:
            report['excluded'][index] = status
    report['answer_parse_status_counts'] = dict(Counter(statuses.values()))
    report['unsupported_labels'] = {a:s for a,s in statuses.items() if s not in {'numeric','symbolic_or_structured'}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    for split in ("train", "validation"):
        table = pq.read_table(args.input_dir / f"{split}.parquet")
        keep, revised, questions = [], [], set()
        for row_index, row in enumerate(table.to_pylist()):
            index = int(row["extra_info"]["index"])
            if index in report["excluded"]: continue
            record = records[index]
            if str(row["reward_model"]["ground_truth"]) != str(record["answer"]):
                raise ValueError(f"Source label mismatch at {split} row {row_index}")
            q = record["problem"].strip()
            questions.add(q)
            if index in report["corrections"]:
                row["reward_model"]["ground_truth"] = report["corrections"][index]["answer"]
                revised.append(index)
            keep.append(row)
        pq.write_table(pa.Table.from_pylist(keep, schema=table.schema), args.output_dir / f"{split}.parquet")
        seen[split] = questions
        report[split] = {"before": table.num_rows, "after": len(keep), "corrected_source_indices": revised}
    if seen["train"] & seen["validation"]:
        raise ValueError("Question groups overlap train and validation")
    report["exclusion_counts"] = dict(Counter(report["excluded"].values()))
    report["source_sha256"] = hashlib.sha256(args.raw_json.read_bytes()).hexdigest()
    report["split_policy"] = "original held-out membership and source order retained"
    report["input_dir"] = str(args.input_dir.resolve())
    report["output_dir"] = str(args.output_dir.resolve())
    (args.output_dir / "quality_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("source_rows", "train", "validation", "exclusion_counts")}), flush=True)
    print(f"Quarantined {len(report['conflicting_groups'])} conflicting groups; retained {len(report['equivalent_label_groups'])} equivalent groups", flush=True)


if __name__ == "__main__":
    main()
