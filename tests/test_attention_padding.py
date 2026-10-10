import os
import unittest
from unittest.mock import patch

import torch

from verl.utils import attention_utils
from verl_patches.attention_padding import install_attention_padding


class AttentionPaddingTest(unittest.TestCase):
    def test_opt_in_padding_round_trip_and_gradients(self):
        original = attention_utils._get_attention_functions
        try:
            with patch.dict(os.environ, {"AGENT_R1_TORCH_PADDING": "1"}), \
                    patch("verl_patches.attention_padding.importlib.util.find_spec", return_value=None):
                self.assertTrue(install_attention_padding())
            values = torch.arange(24.0).reshape(2, 6, 2).requires_grad_()
            mask = torch.tensor([[0, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]])
            packed, indices, lengths, max_length, used = attention_utils.unpad_input(values, mask)
            self.assertTrue(torch.equal(packed, values[mask.bool()]))
            self.assertEqual(lengths.tolist(), [0, 3, 5])
            self.assertEqual(used.tolist(), [3, 2])
            self.assertEqual(max_length, 3)
            restored = attention_utils.pad_input(packed, indices, 2, 6)
            self.assertTrue(torch.equal(restored, values * mask.unsqueeze(-1)))
            restored.sum().backward()
            self.assertTrue(torch.equal(values.grad, mask.unsqueeze(-1).expand_as(values)))
        finally:
            attention_utils._get_attention_functions = original

    def test_fallback_is_disabled_without_opt_in(self):
        original = attention_utils._get_attention_functions
        with patch.dict(os.environ, {"AGENT_R1_TORCH_PADDING": "0"}):
            self.assertFalse(install_attention_padding())
        self.assertIs(attention_utils._get_attention_functions, original)


if __name__ == "__main__":
    unittest.main()
