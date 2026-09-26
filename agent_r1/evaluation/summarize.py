"""Summarize fixed-protocol Eq. (9) audits from one evaluation/checkpoint dump.

python -m agent_r1.evaluation.summarize validation.jsonl --output metrics.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .consistency import PROTOCOL_VERSION


def summarize(entries):
    rows = list(entries)
    if not rows:
        raise ValueError("Cannot evaluate an empty dump")
    checkpoints = {row.get("global_step") for row in rows if "global_step" in row}
    if len(checkpoints) > 1:
        raise ValueError("Select a single checkpoint with --global-step; do not pool training updates")
    outcomes, consistencies, successes, infos = [], [], [], []
    for row in rows:
        info = row["steps"][-1] if row.get("steps") else row
        if info.get("verification_protocol") != PROTOCOL_VERSION:
            raise ValueError("Missing/current-protocol audit required; re-evaluate old dumps rather than infer C_d from rewards")
        checks = info["verification_checks"]
        consistent = int(bool(info["terminal_eligible"]) and bool(checks) and all(float(v) == 1 for v in checks))
        outcome = info.get("acc", info.get("is_correct", info.get("terminal_reward")))
        if outcome not in (0, 1):
            raise ValueError("A binary benchmark outcome is required")
        if consistent != info["verification_consistency"] or float(outcome) * consistent != info["verified_success"]:
            raise ValueError("Audit consistency fields do not match recorded checks/outcome")
        outcomes.append(float(outcome)); consistencies.append(consistent); successes.append(float(outcome) * consistent)
        infos.append(info)
    n = len(rows)
    result = {
        "verification_protocol": PROTOCOL_VERSION, "trajectory_count": n,
        "global_step": next(iter(checkpoints), None), "accuracy": sum(outcomes) / n,
        "C_d_mean": sum(consistencies) / n, "M_d": sum(successes) / n,
        "consistent_count": sum(consistencies), "verified_success_count": sum(successes),
    }
    for key in ("revision_count", "developer_test_count", "retest_count", "revision_after_failed_test"):
        values = [float(info[key]) for info in infos if info.get(key) is not None]
        if values:
            result[key + "_mean"] = sum(values) / len(values)
            result[key + "_denominator"] = len(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--global-step", type=int)
    args = parser.parse_args()
    with args.input.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.global_step is not None:
        rows = [row for row in rows if row.get("global_step") == args.global_step]
    result = json.dumps(summarize(rows), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
