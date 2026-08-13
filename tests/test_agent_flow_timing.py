import unittest
from types import SimpleNamespace

import torch

from agent_r1.agent_flow.agent_flow import AgentFlowManager
from recipes.hotpotqa.hotpotqa_agent_flow import HotpotQAAgentFlow


class AgentFlowTimingTest(unittest.TestCase):
    @staticmethod
    def _output():
        return SimpleNamespace(
            batch={
                "prompts": torch.zeros((3, 2), dtype=torch.long),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 1, 0, 0],
                        [1, 0, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                    ],
                    dtype=torch.long,
                ),
            }
        )

    @staticmethod
    def _manager():
        return object.__new__(AgentFlowManager)

    def test_raw_step_timings_are_aggregated_once_per_trajectory(self):
        metrics = [
            [
                {
                    "generate_sequences": 3.0,
                    "tool_calls": 0.5,
                    "step_generate_sequences": [1.0, 2.0],
                    "step_tool_calls": [0.5, 0.0],
                },
                {
                    "generate_sequences": 4.0,
                    "tool_calls": 1.0,
                    "step_generate_sequences": [4.0],
                    "step_tool_calls": [1.0],
                },
            ]
        ]

        timing = self._manager()._performance_metrics(metrics, [2, 1], self._output())

        self.assertEqual(timing["agent_flow/timing/per_step_available"], 1.0)
        self.assertAlmostEqual(timing["agent_flow/step/generate_sequences/mean"], 7.0 / 3.0)
        self.assertAlmostEqual(timing["agent_flow/step/tool_calls/mean"], 0.5)
        self.assertAlmostEqual(timing["agent_flow/trajectory/generate_sequences/min"], 3.0)
        self.assertAlmostEqual(timing["agent_flow/trajectory/generate_sequences/max"], 4.0)
        self.assertAlmostEqual(timing["agent_flow/trajectory/total/mean"], 4.25)
        self.assertEqual(timing["agent_flow/slowest/num_steps"], 1)
        self.assertEqual(timing["agent_flow/slowest/total_prompt_length"], 2)
        self.assertEqual(timing["agent_flow/slowest/total_response_length"], 3)

    def test_hotpotqa_records_zero_tool_time_for_a_step_without_a_tool_call(self):
        metrics = {
            "generate_sequences": 0.0,
            "tool_calls": 0.0,
            "step_generate_sequences": [],
            "step_tool_calls": [],
        }

        HotpotQAAgentFlow._record_step_timing(metrics, {"generate_sequences": 1.25})

        self.assertEqual(metrics["generate_sequences"], 1.25)
        self.assertEqual(metrics["tool_calls"], 0.0)
        self.assertEqual(metrics["step_generate_sequences"], [1.25])
        self.assertEqual(metrics["step_tool_calls"], [0.0])

    def test_legacy_multistep_metrics_do_not_publish_false_step_statistics(self):
        metrics = [
            [
                {
                    "generate_sequences": 3.0,
                    "tool_calls": 0.5,
                    "step_generate_sequences": [],
                    "step_tool_calls": [],
                },
                {
                    "generate_sequences": 4.0,
                    "tool_calls": 1.0,
                    "step_generate_sequences": [],
                    "step_tool_calls": [],
                },
            ]
        ]

        timing = self._manager()._performance_metrics(metrics, [2, 1], self._output())

        self.assertEqual(timing["agent_flow/timing/per_step_available"], 0.0)
        self.assertNotIn("agent_flow/step/generate_sequences/mean", timing)
        self.assertAlmostEqual(timing["agent_flow/trajectory/generate_sequences/min"], 3.0)
        self.assertAlmostEqual(timing["agent_flow/trajectory/generate_sequences/max"], 4.0)

    def test_misaligned_raw_step_timings_fail_closed(self):
        metrics = [
            [
                {
                    "generate_sequences": 3.0,
                    "tool_calls": 0.5,
                    "step_generate_sequences": [1.0],
                    "step_tool_calls": [0.5, 0.0],
                }
            ]
        ]

        with self.assertRaisesRegex(ValueError, "timing length mismatch"):
            self._manager()._performance_metrics(metrics, [2], self._output())


if __name__ == "__main__":
    unittest.main()
