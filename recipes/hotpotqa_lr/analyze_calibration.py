#!/usr/bin/env python3
"""Fail-closed analysis for the 64-question, n=8 A8-LR calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pvariance
from typing import Any

from recipes.hotpotqa_lr.dsl import DSL_VERSION, REASON_STEP_FORMAT
from recipes.hotpotqa_lr.verifier import VERIFIER_VERSION, trajectory_audit_record, verify_trajectory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = _load_jsonl(args.rollouts)
    if manifest.get("run_mode") != "frozen_policy_calibration":
        raise ValueError("Calibration analyzer requires a frozen_policy_calibration manifest")

    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    local_rewards: list[float] = []
    terminal_rewards: list[float] = []
    replay_matches = 0
    attempted_actions = 0
    valid_actions = 0
    attempted_reason_steps = 0
    parsed_reason_steps = 0
    forbidden_verifier_input_count = 0

    for expected_index, record in enumerate(records):
        if int(record.get("sample_index", -1)) != expected_index:
            raise ValueError("Calibration rows must be a contiguous deterministic prefix")
        question = str(record.get("question") or "")
        transitions = record.get("local_reasoning_transitions")
        if not isinstance(transitions, list):
            transitions = []
        forbidden_verifier_input_count += sum(
            key in transition
            for transition in transitions
            if isinstance(transition, dict)
            for key in ("ground_truth", "gold_answer", "gold_evidence_ids", "supporting_facts")
        )
        offline = trajectory_audit_record(verify_trajectory(transitions, question=question))
        online = record.get("local_reasoning_audit")
        replay_matches += int(isinstance(online, dict) and online == offline)

        local_reward = float(offline["local_reward"])
        terminal_em = float(record.get("score", 0.0))
        local_rewards.append(local_reward)
        terminal_rewards.append(terminal_em)
        group_key = str(record.get("sample_key") or "")
        groups[group_key].append((local_reward, terminal_em))

        turns = int(record.get("num_turns", 0))
        attempted_actions += turns
        searches = record.get("search_queries")
        valid_actions += len(searches) if isinstance(searches, list) else 0
        valid_actions += int(record.get("answer") is not None)
        attempted_reason_steps += len(transitions)
        parsed_reason_steps += int(offline["parsed_reason_step_count"])

    homogeneous_groups = 0
    locally_variable_homogeneous_groups = 0
    group_sizes_valid = True
    for values in groups.values():
        group_sizes_valid &= len(values) == 8
        local_values = [item[0] for item in values]
        terminal_values = [item[1] for item in values]
        if pvariance(terminal_values) <= 1e-12:
            homogeneous_groups += 1
            locally_variable_homogeneous_groups += int(pvariance(local_values) > 1e-12)

    count = len(records)
    parser_validity = valid_actions / attempted_actions if attempted_actions else 0.0
    reason_parseability = parsed_reason_steps / attempted_reason_steps if attempted_reason_steps else 0.0
    replay_equality = replay_matches / count if count else 0.0
    local_positive_rate = sum(value > 0 for value in local_rewards) / count if count else 0.0
    local_saturation_rate = sum(value >= 1.0 - 1e-12 for value in local_rewards) / count if count else 0.0
    conditional_variance_rate = locally_variable_homogeneous_groups / homogeneous_groups if homogeneous_groups else 0.0

    gates = {
        "trajectory_count_512": count == 512,
        "question_groups_64_of_8": len(groups) == 64 and group_sizes_valid,
        "search_finish_parser_validity_ge_0_95": parser_validity >= 0.95,
        "reason_step_parseability_ge_0_90": reason_parseability >= 0.90,
        "online_offline_replay_equality_1_0": replay_equality == 1.0,
        "local_positive_rate_between_0_05_and_0_90": 0.05 <= local_positive_rate <= 0.90,
        "conditional_local_variance_rate_ge_0_20": conditional_variance_rate >= 0.20,
        "local_reward_saturation_lt_0_90": local_saturation_rate < 0.90,
        "gold_inputs_visible_to_verifier_zero": forbidden_verifier_input_count == 0,
    }
    code_hashes = manifest.get("code_sha256")
    if not isinstance(code_hashes, dict):
        code_hashes = {}
    report = {
        "contract_version": "hotpotqa-a8-lr-calibration-report-v1",
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": {
            "trajectory_count": count,
            "question_group_count": len(groups),
            "search_finish_parser_validity": parser_validity,
            "reason_step_parseability": reason_parseability,
            "online_offline_replay_equality": replay_equality,
            "local_positive_rate": local_positive_rate,
            "local_reward_mean": fmean(local_rewards) if local_rewards else 0.0,
            "local_reward_saturation_rate": local_saturation_rate,
            "terminal_em_mean": fmean(terminal_rewards) if terminal_rewards else 0.0,
            "em_homogeneous_group_count": homogeneous_groups,
            "locally_variable_em_homogeneous_group_count": locally_variable_homogeneous_groups,
            "conditional_local_variance_rate": conditional_variance_rate,
            "forbidden_verifier_input_count": forbidden_verifier_input_count,
        },
        "offline_reward_replay": {
            "lr30_mean": (
                fmean(0.7 * em + 0.3 * local for local, em in zip(local_rewards, terminal_rewards)) if records else 0.0
            ),
            "same_interface_base_mean": fmean(terminal_rewards) if terminal_rewards else 0.0,
            "duplicate_base_generation_required": False,
        },
        "reason_step_format": REASON_STEP_FORMAT,
        "dsl_version": DSL_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "seed": manifest.get("scientific_config", {}).get("calibration_seed"),
        "model": {
            "path": manifest.get("model", {}).get("path"),
            "identity_sha256": manifest.get("model", {}).get("identity_sha256"),
        },
        "code_sha256": code_hashes,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": _sha256(args.manifest),
        "source_rollouts": str(args.rollouts.resolve()),
        "source_rollouts_sha256": _sha256(args.rollouts),
    }
    _write_json_atomic(args.output, report)
    print(json.dumps({"passed": report["passed"], "report": str(args.output)}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
