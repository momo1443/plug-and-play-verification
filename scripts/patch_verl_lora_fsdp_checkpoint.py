#!/usr/bin/env python3
"""Make veRL FSDP checkpoint CPU offload configurable for LoRA training.

FSDP's CPU-offloaded sharded state dict conflicts with CUDA-resident PEFT LoRA
parameters during ``model.state_dict()``. Set
``VERL_FSDP_CHECKPOINT_OFFLOAD_TO_CPU=0`` for LoRA runs so both tensor groups
stay on CUDA while the checkpoint state dict is assembled. The default remains
the upstream CPU-offload behavior for other runs.
"""

import os
import sys


TARGET = os.path.join(
    os.path.dirname(sys.executable),
    "..",
    "lib",
    "python3.10",
    "site-packages",
    "verl",
    "utils",
    "checkpoint",
    "fsdp_checkpoint_manager.py",
)
TARGET = os.path.abspath(TARGET)

OLD_EXPR = "True if is_cuda_available else False"
NEW_EXPR = "_checkpoint_offload_to_cpu()"
PATCHED_CALL = f"offload_to_cpu={NEW_EXPR}"
HELPER = '''\n\ndef _checkpoint_offload_to_cpu() -> bool:\n    """Return the configured FSDP checkpoint state-dict offload policy."""\n    configured = os.getenv("VERL_FSDP_CHECKPOINT_OFFLOAD_TO_CPU", "1").lower()\n    return is_cuda_available and configured not in {"0", "false", "no", "off"}\n'''


def main() -> None:
    if not os.path.exists(TARGET):
        print(f"ERROR: Target file not found: {TARGET}")
        sys.exit(1)

    with open(TARGET, encoding="utf-8") as file:
        content = file.read()

    if "def _checkpoint_offload_to_cpu()" in content:
        if content.count(PATCHED_CALL) != 4:
            print("ERROR: Existing checkpoint-offload patch has an unexpected number of call sites.")
            sys.exit(1)
        print(f"Checkpoint offload patch already applied: {TARGET}")
        return

    expected_occurrences = 4
    actual_occurrences = content.count(OLD_EXPR)
    if actual_occurrences != expected_occurrences:
        print(
            "ERROR: Expected "
            f"{expected_occurrences} CPU-offload expressions, found {actual_occurrences}: {TARGET}"
        )
        sys.exit(1)

    marker = 'logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))'
    if marker not in content:
        print(f"ERROR: Could not find logger setup marker: {TARGET}")
        sys.exit(1)

    content = content.replace(marker, marker + HELPER, 1)
    content = content.replace(OLD_EXPR, NEW_EXPR)

    with open(TARGET, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"Applied configurable checkpoint-offload patch: {TARGET}")


if __name__ == "__main__":
    main()
