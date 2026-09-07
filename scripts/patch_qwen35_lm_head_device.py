#!/usr/bin/env python3
"""Patch verl's qwen3_5 model forward_with_triton_backend and forward_with_fused_kernel_backend
to handle FSDP2 CPUOffloadPolicy where self.lm_head.weight may be on CPU.

When FSDP2 + CPUOffloadPolicy is used, parameters are offloaded to CPU between
forward/backward passes. The lm_head.weight ends up on CPU, but the triton and
fused kernels require all tensors to be on CUDA.

This patch:
1. Moves the DTensor check before the dtype conversion
2. Adds a CPU→CUDA device transfer for lm_head.weight when offloaded

Applied to: verl/models/transformers/qwen3_5.py
"""

import sys

TARGET_FILE = "/nas/deepresearch/conda/envs/agenticrl/lib/python3.10/site-packages/verl/models/transformers/qwen3_5.py"

# Patch 1: forward_with_triton_backend
OLD_TRITON = """    vocab_weights = self.lm_head.weight
    hidden_states = hidden_states.to(vocab_weights.dtype)
    if isinstance(vocab_weights, DTensor):
        vocab_weights = vocab_weights.full_tensor()

    log_probs, entropy = linear_cross_entropy("""

NEW_TRITON = """    vocab_weights = self.lm_head.weight
    if isinstance(vocab_weights, DTensor):
        vocab_weights = vocab_weights.full_tensor()
    # FSDP2 CPUOffloadPolicy may leave lm_head.weight on CPU;
    # move it to the same device as hidden_states before calling kernels.
    if vocab_weights.device.type == "cpu":
        vocab_weights = vocab_weights.to(hidden_states.device, non_blocking=True)
    hidden_states = hidden_states.to(vocab_weights.dtype)

    log_probs, entropy = linear_cross_entropy("""

# Patch 2: forward_with_fused_kernel_backend
OLD_FUSED = """    fused_linear_for_ppo = FusedLinearForPPO()
    vocab_weights = self.lm_head.weight
    if isinstance(vocab_weights, DTensor):
        vocab_weights = vocab_weights.full_tensor()

    ulysses_sequence_parallel_size = get_ulysses_sequence_parallel_world_size()"""

NEW_FUSED = """    fused_linear_for_ppo = FusedLinearForPPO()
    vocab_weights = self.lm_head.weight
    if isinstance(vocab_weights, DTensor):
        vocab_weights = vocab_weights.full_tensor()
    # FSDP2 CPUOffloadPolicy may leave lm_head.weight on CPU;
    # move it to the same device as hidden_states before calling kernels.
    if vocab_weights.device.type == "cpu":
        vocab_weights = vocab_weights.to(hidden_states.device, non_blocking=True)

    ulysses_sequence_parallel_size = get_ulysses_sequence_parallel_world_size()"""


def patch_file():
    with open(TARGET_FILE, "r") as f:
        content = f.read()

    changed = False

    if OLD_TRITON in content:
        content = content.replace(OLD_TRITON, NEW_TRITON, 1)
        print("Patched forward_with_triton_backend.")
        changed = True
    elif "FSDP2 CPUOffloadPolicy may leave lm_head.weight on CPU" in content:
        print("Already patched (triton backend). Skipping.")
    else:
        print("ERROR: Could not find original triton backend pattern!")
        sys.exit(1)

    if OLD_FUSED in content:
        content = content.replace(OLD_FUSED, NEW_FUSED, 1)
        print("Patched forward_with_fused_kernel_backend.")
        changed = True
    elif "FSDP2 CPUOffloadPolicy may leave lm_head.weight on CPU" in content:
        print("Already patched (fused kernel backend). Skipping.")
    else:
        print("ERROR: Could not find original fused kernel backend pattern!")
        sys.exit(1)

    if changed:
        with open(TARGET_FILE, "w") as f:
            f.write(content)
        print(f"Saved patches to {TARGET_FILE}")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    patch_file()
