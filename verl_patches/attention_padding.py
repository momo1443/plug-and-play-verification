"""Opt-in Torch padding helpers for veRL's SDPA training profile.

veRL imports FlashAttention to pack/unpack tokens even when the model uses
SDPA. Its NPU helper module already implements these operations in ordinary
PyTorch; the same device-independent functions work on CPU and CUDA.
This fallback only changes padding helpers, never the attention kernel.
"""

import importlib.util
import os


def install_attention_padding() -> bool:
    if os.environ.get("AGENT_R1_TORCH_PADDING", "0") != "1":
        return False
    if importlib.util.find_spec("flash_attn") is not None:
        return False

    from verl.utils import attention_utils
    from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input

    def torch_padding_functions():
        return index_first_axis, pad_input, rearrange, unpad_input

    attention_utils._get_attention_functions = torch_padding_functions
    return True
