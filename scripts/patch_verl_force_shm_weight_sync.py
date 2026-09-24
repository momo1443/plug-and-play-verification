#!/usr/bin/env python3
"""Force shared-memory vLLM weight synchronization on this host."""

import os
import sys


target = os.path.join(
    os.path.dirname(sys.executable),
    "..",
    "lib",
    "python3.10",
    "site-packages",
    "verl",
    "workers",
    "rollout",
    "vllm_rollout",
    "vllm_rollout.py",
)
target = os.path.abspath(target)
old = "        self.use_shm = not is_support_ipc()"
env_override = '        self.use_shm = os.environ.get("VERL_FORCE_WEIGHT_SYNC_SHM", "").lower() in {"1", "true", "yes"} or not is_support_ipc()'
new = "        self.use_shm = True  # CUDA IPC stalls during 9B FSDP2 CPU-offload synchronization on this host."

with open(target) as file:
    content = file.read()

if new in content:
    print("Shared-memory weight-sync override already applied.")
elif old in content or env_override in content:
    with open(target, "w") as file:
        file.write(content.replace(old, new, 1).replace(env_override, new, 1))
    print(f"Applied shared-memory weight-sync override: {target}")
else:
    print(f"ERROR: expected vLLM weight-sync assignment was not found: {target}")
    sys.exit(1)
