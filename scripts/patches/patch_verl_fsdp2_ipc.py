"""Patch verl's rebuild_ipc to handle FSDP2 + CPUOffloadPolicy CPU tensor handles.

The bug: rebuild_ipc assumes CUDA IPC handle format (15 args) and blindly
sets list_args[6] = device_id, but CPU tensor handles only have 3 args.

The fix: Only set device_id when args list is long enough (CUDA handles),
and move CPU tensors to GPU after rebuild.
"""
import sys
import os

TARGET = os.path.join(
    os.path.dirname(sys.executable), "..", "lib", "python3.10", "site-packages",
    "verl", "workers", "rollout", "vllm_rollout", "bucketed_weight_transfer.py"
)
TARGET = os.path.abspath(TARGET)

OLD_CODE = '''def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer'''

NEW_CODE = '''def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        # FSDP2 + CPUOffloadPolicy may produce CPU tensor handles with fewer args
        # (3 args for CPU vs 15 for CUDA); only patch device_id for CUDA handles
        if len(list_args) > 6:
            list_args[6] = device_id
    buffer = func(*list_args)
    # If the buffer landed on CPU (FSDP2 CPUOffloadPolicy path), move it to the
    # target GPU so vLLM can use it.
    if device_id is not None and buffer.device.type == "cpu":
        buffer = buffer.to(f"cuda:{device_id}", non_blocking=True)
    return buffer'''

if not os.path.exists(TARGET):
    print(f"ERROR: Target file not found: {TARGET}")
    sys.exit(1)

with open(TARGET, 'r') as f:
    content = f.read()

if NEW_CODE in content:
    print("Patch already applied.")
    sys.exit(0)

if OLD_CODE not in content:
    print("ERROR: Could not find the code to patch. File may have been updated.")
    sys.exit(1)

content = content.replace(OLD_CODE, NEW_CODE)
with open(TARGET, 'w') as f:
    f.write(content)

print(f"Successfully patched: {TARGET}")
