import unittest

import numpy as np
import torch
from tensordict import TensorDict

from agent_r1.agent_flow.agent_flow import _align_agent_flow_output_schema
from verl.protocol import DataProto


def _output(*, reward_fields: dict[str, object], reward_extra_keys: list[str]) -> DataProto:
    return DataProto(
        batch=TensorDict({"value": torch.tensor([1])}, batch_size=1),
        non_tensor_batch={
            "trajectory_uids": np.array(["trajectory"], dtype=object),
            **{key: np.array([value], dtype=object) for key, value in reward_fields.items()},
        },
        meta_info={"metrics": [], "reward_extra_keys": reward_extra_keys},
    )


class AgentFlowOutputSchemaTest(unittest.TestCase):
    def test_optional_reward_field_is_filled_across_worker_chunks(self):
        field = "optimizer_answer_reachable_credit"
        without_credit = _output(reward_fields={}, reward_extra_keys=[])
        with_credit = _output(
            reward_fields={field: 1.0},
            reward_extra_keys=[field],
        )

        reward_extra_keys = _align_agent_flow_output_schema([without_credit, with_credit])
        combined = DataProto.concat([without_credit, with_credit])

        self.assertEqual(reward_extra_keys, [field])
        self.assertEqual(without_credit.meta_info["reward_extra_keys"], [field])
        self.assertEqual(with_credit.meta_info["reward_extra_keys"], [field])
        self.assertEqual(combined.non_tensor_batch[field].tolist(), [None, 1.0])

    def test_union_covers_optional_non_reward_fields(self):
        without_optional = _output(reward_fields={}, reward_extra_keys=[])
        with_optional = _output(
            reward_fields={"multi_modal_inputs": {"image": "payload"}},
            reward_extra_keys=[],
        )

        _align_agent_flow_output_schema([without_optional, with_optional])
        combined = DataProto.concat([without_optional, with_optional])

        self.assertEqual(
            combined.non_tensor_batch["multi_modal_inputs"].tolist(),
            [None, {"image": "payload"}],
        )


if __name__ == "__main__":
    unittest.main()
