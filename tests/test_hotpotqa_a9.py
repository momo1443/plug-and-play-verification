"""Tests for A9 certificate-grounded RLVR components."""

import dataclasses
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from recipes.hotpotqa_a9.dsl import (
    FinishCertificate,
    SearchCertificate,
    parse_finish_certificate,
    parse_search_certificate,
)
from recipes.hotpotqa_a9.protocol import (
    parse_finish_call,
    parse_search_call,
)
from recipes.hotpotqa_a9.reward_contract import (
    CONTRACT_CERT_MIX,
    CONTRACT_FORMAT_STRICT,
)
from recipes.hotpotqa_a9.agent_flow import optimizer_reward_schedule
from recipes.hotpotqa_a9.verifier import (
    trajectory_audit_record,
    verify_coupling,
    verify_grounding,
    verify_process,
    verify_trajectory,
)
from recipes.hotpotqa.reward_arm import (
    RewardArm,
    search_step_reward,
    training_reward_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Tool-call XML delimiters matching the LR protocol regex.
# Built with chr() to avoid encoding issues with angle brackets in tooling.
_TOOL_OPEN = chr(60) + "tool_call" + chr(62)
_TOOL_CLOSE = chr(60) + "/tool_call" + chr(62)


# -- Certificate Parsing --


class CertificateParsingTest(unittest.TestCase):
    def test_search_certificate_valid_fields(self):
        cert, errors = parse_search_certificate(
            {"source_id": "passage:17", "support_span": "Henry Miller married June Miller", "target": "June Miller"}
        )
        self.assertIsNotNone(cert)
        self.assertEqual(errors, ())
        self.assertEqual(cert.source_id, "passage:17")
        self.assertEqual(cert.support_span, "Henry Miller married June Miller")
        self.assertEqual(cert.target, "June Miller")

    def test_search_certificate_missing_field(self):
        cert, errors = parse_search_certificate(
            {"source_id": "passage:17", "support_span": "some text"}
        )
        self.assertIsNotNone(cert)
        self.assertIn("certificate_fields_invalid", errors)
        self.assertIn("target_invalid", errors)

    def test_search_certificate_extra_field(self):
        cert, errors = parse_search_certificate(
            {"source_id": "passage:17", "support_span": "s", "target": "t", "extra": 1}
        )
        self.assertIsNotNone(cert)
        self.assertIn("certificate_fields_invalid", errors)

    def test_search_certificate_not_object(self):
        cert, errors = parse_search_certificate("not a dict")
        self.assertIsNone(cert)
        self.assertIn("certificate_not_object", errors)

    def test_search_certificate_empty_source_id(self):
        cert, errors = parse_search_certificate(
            {"source_id": "", "support_span": "s", "target": "t"}
        )
        self.assertIsNone(cert)
        self.assertIn("source_id_invalid", errors)

    def test_finish_certificate_valid_fields(self):
        cert, errors = parse_finish_certificate(
            {"source_id": "passage:42", "support_span": "June Miller was an American", "answer_span": "American"}
        )
        self.assertIsNotNone(cert)
        self.assertEqual(errors, ())
        self.assertEqual(cert.source_id, "passage:42")
        self.assertEqual(cert.answer_span, "American")

    def test_finish_certificate_missing_answer_span(self):
        cert, errors = parse_finish_certificate(
            {"source_id": "passage:42", "support_span": "some text"}
        )
        self.assertIsNotNone(cert)
        self.assertIn("certificate_fields_invalid", errors)
        self.assertIn("answer_span_invalid", errors)


# -- Protocol Parsing --


class ProtocolTest(unittest.TestCase):
    def _make_search_json(self, query, cert=None):
        args = {"query": query}
        if cert is not None:
            args["certificate"] = cert
        inner = json.dumps({"name": "search", "arguments": args})
        return _TOOL_OPEN + "\n" + inner + "\n" + _TOOL_CLOSE

    def _make_finish_json(self, answer, cert=None):
        args = {"status": "answer", "answer": answer}
        if cert is not None:
            args["certificate"] = cert
        inner = json.dumps({"name": "finish", "arguments": args})
        return _TOOL_OPEN + "\n" + inner + "\n" + _TOOL_CLOSE

    def test_search_call_parse_with_valid_certificate(self):
        text = self._make_search_json(
            "June Miller nationality",
            {"source_id": "passage:17", "support_span": "Henry Miller married June Miller", "target": "June Miller"}
        )
        audit = parse_search_call(text)
        self.assertIsNotNone(audit.query)
        self.assertEqual(audit.query, "June Miller nationality")
        self.assertIsNotNone(audit.certificate)

    def test_search_call_parse_with_invalid_certificate_still_extracts_query(self):
        text = self._make_search_json(
            "June Miller nationality",
            {"bad": "field"}
        )
        audit = parse_search_call(text)
        self.assertIsNotNone(audit.query)
        self.assertEqual(audit.query, "June Miller nationality")
        self.assertIsNone(audit.certificate)

    def test_finish_call_parse_with_invalid_certificate_still_extracts_answer(self):
        text = self._make_finish_json(
            "American",
            {"bad": "field"}
        )
        audit = parse_finish_call(text)
        self.assertIsNotNone(audit.answer)
        self.assertEqual(audit.answer, "American")
        self.assertIsNone(audit.certificate)

    def test_unparseable_call_returns_none_query(self):
        audit = parse_search_call("some random text without tool calls")
        self.assertIsNone(audit.query)


# -- Verifier --


class VerifierTest(unittest.TestCase):
    def _make_artifacts(self):
        return {
            "passage:17": {
                "artifact_id": "passage:17",
                "passage_id": "17",
                "title": "Henry Miller",
                "text": "Henry Miller married June Miller in 1924. She was an American writer.",
                "content_sha256": None,
            },
        }

    def setUp(self):
        from recipes.hotpotqa_lr.verifier import artifact_content_sha256
        self.artifacts = self._make_artifacts()
        self.artifacts["passage:17"]["content_sha256"] = artifact_content_sha256(
            self.artifacts["passage:17"]["text"]
        )

    def test_grounding_valid_when_span_in_artifact(self):
        cert = SearchCertificate(source_id="passage:17", support_span="Henry Miller married June Miller", target="June Miller")
        self.assertTrue(verify_grounding(cert, self.artifacts))

    def test_grounding_invalid_when_span_not_in_artifact(self):
        cert = SearchCertificate(source_id="passage:17", support_span="This text does not appear", target="June Miller")
        self.assertFalse(verify_grounding(cert, self.artifacts))

    def test_grounding_invalid_when_source_not_found(self):
        cert = SearchCertificate(source_id="passage:999", support_span="anything", target="anything")
        self.assertFalse(verify_grounding(cert, self.artifacts))

    def test_coupling_search_target_matches_query(self):
        cert = SearchCertificate(source_id="passage:17", support_span="Henry Miller married June Miller", target="June Miller")
        self.assertTrue(verify_coupling("search", "June Miller nationality", cert))

    def test_coupling_search_target_not_in_query(self):
        cert = SearchCertificate(source_id="passage:17", support_span="Henry Miller married June Miller", target="Eleanor")
        self.assertFalse(verify_coupling("search", "June Miller nationality", cert))

    def test_coupling_finish_answer_matches(self):
        cert = FinishCertificate(source_id="passage:17", support_span="She was an American writer", answer_span="American")
        self.assertTrue(verify_coupling("finish", "American", cert))

    def test_coupling_finish_answer_mismatch(self):
        cert = FinishCertificate(source_id="passage:17", support_span="She was an American writer", answer_span="British")
        self.assertFalse(verify_coupling("finish", "American", cert))

    def test_trajectory_credit_with_all_valid(self):
        transitions = [{
            "certificate": {"source_id": "passage:17", "support_span": "Henry Miller married June Miller", "target": "June Miller"},
            "certificate_parsed": {"source_id": "passage:17", "support_span": "Henry Miller married June Miller", "target": "June Miller"},
            "action_type": "search",
            "action_value": "June Miller nationality",
            "available_artifacts": self.artifacts,
        }]
        audits = verify_trajectory(transitions, reward_horizon=3)
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0].grounding_valid, 1)
        self.assertEqual(audits[0].coupling_valid, 1)
        self.assertEqual(audits[0].own_valid, 1)
        self.assertAlmostEqual(audits[0].raw_local_credit, 1.0 / 3.0)

        result = verify_process(transitions, [1], reward_horizon=3)
        self.assertAlmostEqual(result.process_reward, 1.0 / 3.0)
        self.assertEqual(result.components_by_step(), {2: 1.0 / 3.0})

    def test_trajectory_zero_credit_on_parse_failure(self):
        transitions = [{
            "certificate": {"bad": "field"},
            "certificate_parsed": None,
            "action_type": "search",
            "action_value": "June Miller nationality",
            "available_artifacts": self.artifacts,
        }]
        audits = verify_trajectory(transitions, reward_horizon=3)
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0].own_valid, 0)
        self.assertEqual(audits[0].raw_local_credit, 0.0)

    def test_trajectory_audit_record(self):
        transitions = [{
            "certificate": {"source_id": "passage:17", "support_span": "Henry Miller married June Miller", "target": "June Miller"},
            "certificate_parsed": {"source_id": "passage:17", "support_span": "Henry Miller married June Miller", "target": "June Miller"},
            "action_type": "search",
            "action_value": "June Miller nationality",
            "available_artifacts": self.artifacts,
        }]
        audits = verify_trajectory(transitions, reward_horizon=3)
        record = trajectory_audit_record(audits)
        self.assertAlmostEqual(record["local_reward"], 1.0 / 3.0)
        self.assertEqual(record["parsed_certificate_count"], 1)
        self.assertEqual(record["credited_transition_count"], 1)

    def test_trajectory_horizon_exceeded(self):
        base = {
            "certificate": {"source_id": "passage:17", "support_span": "Henry Miller married June Miller", "target": "June Miller"},
            "certificate_parsed": {"source_id": "passage:17", "support_span": "Henry Miller married June Miller", "target": "June Miller"},
            "action_type": "search",
            "action_value": "June Miller nationality",
            "available_artifacts": self.artifacts,
        }
        transitions = [dict(base) for _ in range(4)]
        audits = verify_trajectory(transitions, reward_horizon=3)
        self.assertEqual(len(audits), 4)
        for i in range(3):
            self.assertGreater(audits[i].raw_local_credit, 0)
        self.assertEqual(audits[3].raw_local_credit, 0.0)
        self.assertIn("reward_horizon_exceeded", audits[3].errors)


# -- Reward Contract --


class RewardContractTest(unittest.TestCase):
    def test_cert_mix_weights(self):
        self.assertAlmostEqual(CONTRACT_CERT_MIX.terminal_weight, 0.5)
        self.assertAlmostEqual(CONTRACT_CERT_MIX.process_weight, 0.5)

    def test_cert_mix_weights_sum(self):
        self.assertAlmostEqual(CONTRACT_CERT_MIX.terminal_weight + CONTRACT_CERT_MIX.process_weight, 1.0, places=10)

    def test_cert_mix_format_gate_off(self):
        self.assertFalse(CONTRACT_CERT_MIX.format_gate)

    def test_format_strict_weights(self):
        self.assertAlmostEqual(CONTRACT_FORMAT_STRICT.terminal_weight, 0.5)
        self.assertAlmostEqual(CONTRACT_FORMAT_STRICT.process_weight, 0.5)

    def test_format_strict_format_gate_on(self):
        self.assertTrue(CONTRACT_FORMAT_STRICT.format_gate)


# -- Reward Schedule --


class RewardScheduleTest(unittest.TestCase):
    def test_validation_returns_deterministic(self):
        tw, pw, phase = optimizer_reward_schedule(
            global_step=200,
            is_validation=True,
            em_warmup_steps=50,
            prompt_group_key="validation:7",
        )
        self.assertAlmostEqual(tw, 1.0)
        self.assertAlmostEqual(pw, 0.0)
        self.assertEqual(phase, "validation_terminal_em")

    def test_warmup_returns_deterministic(self):
        tw, pw, phase = optimizer_reward_schedule(
            global_step=50,
            is_validation=False,
            em_warmup_steps=50,
            prompt_group_key="train:7",
        )
        self.assertAlmostEqual(tw, 1.0)
        self.assertAlmostEqual(pw, 0.0)
        self.assertEqual(phase, "em_warmup")

    def test_certificate_uniform_without_format_gate(self):
        """Prompt-group hashes retain the intended uniform mixture distribution."""
        terminals = []
        for prompt_index in range(1000):
            tw, pw, phase = optimizer_reward_schedule(
                global_step=200,
                is_validation=False,
                em_warmup_steps=50,
                prompt_group_key=f"train:{prompt_index}",
            )
            self.assertEqual(phase, "certificate_uniform")
            self.assertAlmostEqual(tw + pw, 1.0, places=10)
            self.assertGreaterEqual(tw, 0.0)
            self.assertLessEqual(tw, 1.0)
            terminals.append(tw)
        mean_tw = sum(terminals) / len(terminals)
        self.assertAlmostEqual(mean_tw, 0.5, delta=0.05)

    def test_rollouts_in_same_prompt_group_share_weight(self):
        weights = [
            optimizer_reward_schedule(
                global_step=200,
                is_validation=False,
                em_warmup_steps=50,
                prompt_group_key="train:19",
            )[:2]
            for _ in range(4)
        ]
        self.assertEqual(len(set(weights)), 1)

    def test_weight_changes_across_prompt_groups_and_updates(self):
        group_a = optimizer_reward_schedule(
            global_step=200,
            is_validation=False,
            em_warmup_steps=50,
            prompt_group_key="train:19",
        )[0]
        group_b = optimizer_reward_schedule(
            global_step=200,
            is_validation=False,
            em_warmup_steps=50,
            prompt_group_key="train:20",
        )[0]
        next_update = optimizer_reward_schedule(
            global_step=201,
            is_validation=False,
            em_warmup_steps=50,
            prompt_group_key="train:19",
        )[0]
        self.assertNotEqual(group_a, group_b)
        self.assertNotEqual(group_a, next_update)

    def test_no_warmup_goes_straight_to_uniform(self):
        tw, pw, phase = optimizer_reward_schedule(
            global_step=1,
            is_validation=False,
            em_warmup_steps=0,
            prompt_group_key="train:7",
        )
        self.assertEqual(phase, "certificate_uniform")
        self.assertAlmostEqual(tw + pw, 1.0, places=10)


# -- Format Gate --


class FormatGateTest(unittest.TestCase):
    """Tests for the format_gate mechanism that zeros all trajectory reward
    when the model fails to produce a valid finish tool call."""

    def test_format_gate_default_off(self):
        """By default, format_gate is disabled."""
        self.assertFalse(CONTRACT_CERT_MIX.format_gate)

    def test_format_gate_strict_on(self):
        """CONTRACT_FORMAT_STRICT has format_gate enabled."""
        self.assertTrue(CONTRACT_FORMAT_STRICT.format_gate)

    def test_format_gate_warmup_exempt(self):
        """During EM warmup, format_gate should not trigger regardless of
        finish validity.  Verify via optimizer_reward_schedule returning
        'em_warmup' phase which the gate checks."""
        tw, pw, phase = optimizer_reward_schedule(
            global_step=50,
            is_validation=False,
            em_warmup_steps=50,
            prompt_group_key="train:7",
        )
        self.assertEqual(phase, "em_warmup")

    def test_format_gate_post_warmup_phase(self):
        """The strict ablation preserves the shared uniform A9 schedule."""
        tw, pw, phase = optimizer_reward_schedule(
            global_step=200,
            is_validation=False,
            em_warmup_steps=50,
            prompt_group_key="train:7",
        )
        self.assertEqual(phase, "certificate_uniform")


# -- Reward Arm (reward_arm.py) --


class RewardArmTest(unittest.TestCase):
    def test_a9_contract_updated(self):
        contract = training_reward_contract(RewardArm.A9)
        self.assertEqual(dataclasses.asdict(contract), {
            "process_weight": 0.5,
            "terminal_weight": 0.5,
            "final_response_mask": 1,
        })

    def test_a9_subarm_contracts(self):
        contract = training_reward_contract(RewardArm.A9_CERT_MIX)
        self.assertAlmostEqual(contract.terminal_weight, 0.5)
        self.assertAlmostEqual(contract.process_weight, 0.5)

    def test_a9_no_binary_check(self):
        result = search_step_reward(RewardArm.A9_CERT_MIX, 0.33, is_validation=False)
        self.assertAlmostEqual(result, 0.33 * 0.5)# -- Validation-time Lenient Fallback --


class A9LenientFallbackTest(unittest.TestCase):
    """Tests for A9 compute_score falling back to <answer> tags during
    validation while remaining strict during training."""

    def test_validation_lenient_fallback_answer_tag(self):
        """When no finish tool call is present but <answer> tag is,
        validation-time scoring should extract the answer leniently."""
        from recipes.hotpotqa_a9.reward_fn import compute_score as a9_compute_score

        completion = "<answer>American</answer>"
        extra_info = {"_agent_r1_is_validation": True}
        self.assertEqual(
            a9_compute_score("hotpotqa_distractor", completion, "American", extra_info=extra_info),
            1.0,
        )

    def test_training_stays_strict_without_validation_flag(self):
        """Without _agent_r1_is_validation=True, the strict behaviour is
        preserved: a missing finish tool call scores 0."""
        from recipes.hotpotqa_a9.reward_fn import compute_score as a9_compute_score

        completion = "<answer>American</answer>"
        # No extra_info -> training mode
        self.assertEqual(
            a9_compute_score("hotpotqa_distractor", completion, "American"),
            0.0,
        )
        # Explicit is_validation=False
        self.assertEqual(
            a9_compute_score(
                "hotpotqa_distractor",
                completion,
                "American",
                extra_info={"_agent_r1_is_validation": False},
            ),
            0.0,
        )

    def test_validation_prefers_finish_answer_over_tag(self):
        """When a valid finish tool call is present, the answer inside it
        should be used even during validation (not the <answer> tag)."""
        from recipes.hotpotqa_a9.reward_fn import compute_score as a9_compute_score

        _TOOL_OPEN = chr(60) + "tool_call" + chr(62)
        _TOOL_CLOSE = chr(60) + "/tool_call" + chr(62)
        completion = (
            f"{_TOOL_OPEN}\n"
            '{"name": "finish", "arguments": {"status": "answer", "answer": "American", '
            '"certificate": {"source_id": "passage:1", "support_span": "test", '
            '"answer_span": "American"}}}\n'
            f"{_TOOL_CLOSE}"
        )
        extra_info = {"_agent_r1_is_validation": True}
        self.assertEqual(
            a9_compute_score("hotpotqa_distractor", completion, "American", extra_info=extra_info),
            1.0,
        )

    def test_validation_lenient_fallback_wrong_answer(self):
        """Lenient fallback should still return 0 for wrong answers."""
        from recipes.hotpotqa_a9.reward_fn import compute_score as a9_compute_score

        completion = "<answer>British</answer>"
        extra_info = {"_agent_r1_is_validation": True}
        self.assertEqual(
            a9_compute_score("hotpotqa_distractor", completion, "American", extra_info=extra_info),
            0.0,
        )


# -- Launcher Integration --


class LauncherTest(unittest.TestCase):
    def test_rlvr_sh_routes_a9_to_certificate_agent(self):
        shared = (PROJECT_ROOT / "examples/hotpotqa/run_rlvr.sh").read_text(encoding="utf-8")
        self.assertIn("recipes/hotpotqa_a9/base.yaml", shared)
        self.assertIn("hotpotqa_certificate_agent", shared)
        self.assertIn("recipes/hotpotqa_a9/reward_fn.py", shared)
        self.assertIn("A9_CERT_MIX", shared)

    def test_a9_launcher_sets_arm_env(self):
        launcher = (PROJECT_ROOT / "examples/hotpotqa/run_a9.sh").read_text(encoding="utf-8")
        self.assertIn("HOTPOTQA_REWARD_ARM=A9_CERT_MIX", launcher)
        self.assertIn("HOTPOTQA_A9_EM_WARMUP_STEPS", launcher)


if __name__ == "__main__":
    unittest.main()
