"""Patch Qwen3_5 rotary embeddings to handle FSDP2 + CPUOffloadPolicy device mismatch.

The bug: FSDP2 + CPUOffloadPolicy keeps register_buffer tensors (inv_freq) on CPU
while computation tensors (position_ids) are on GPU. This causes:
  RuntimeError: Expected all tensors to be on the same device

The fix: Move inv_freq to position_ids.device before computation in forward().
"""
import sys
import os

TARGET = os.path.join(
    os.path.dirname(sys.executable), "..", "lib", "python3.10", "site-packages",
    "transformers", "models", "qwen3_5", "modeling_qwen3_5.py"
)
TARGET = os.path.abspath(TARGET)

# Patch 1: Qwen3_5VisionRotaryEmbedding.forward
OLD_VISION = '''    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)'''

NEW_VISION = '''    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        inv_freq = self.inv_freq.to(position_ids.device)
        return (position_ids.unsqueeze(-1) * inv_freq).flatten(1)'''

# Patch 2: Qwen3_5TextRotaryEmbedding - need to find the forward method
# Let me search for it more carefully
if not os.path.exists(TARGET):
    print(f"ERROR: Target file not found: {TARGET}")
    sys.exit(1)

with open(TARGET, 'r') as f:
    content = f.read()

if NEW_VISION in content:
    print("Patch already applied.")
    sys.exit(0)

if OLD_VISION not in content:
    print("ERROR: Could not find VisionRotaryEmbedding.forward code to patch.")
    sys.exit(1)

content = content.replace(OLD_VISION, NEW_VISION)

# Now patch the TextRotaryEmbedding - look for its forward method
# The TextRotaryEmbedding.forward uses self.inv_freq too
# Need to find the exact pattern
import re

# Find the forward method that uses self.inv_freq in TextRotaryEmbedding
# Pattern: freqs = (position_ids.unsqueeze(-1) * self.inv_freq)
text_rope_patterns = [
    # Common pattern in HuggingFace RoPE implementations
    ("freqs = (position_ids.unsqueeze(-1) * self.inv_freq)",
     "inv_freq = self.inv_freq.to(position_ids.device)\n        freqs = (position_ids.unsqueeze(-1) * inv_freq)"),
]

patched = False
for old, new in text_rope_patterns:
    if old in content:
        content = content.replace(old, new)
        patched = True
        break

if not patched:
    # Save what we have (vision patch only) and report
    print("WARNING: Could not find TextRotaryEmbedding forward pattern. Applied vision patch only.")
    # Let's look for any remaining self.inv_freq usage in forward methods
    remaining = content.count("self.inv_freq")
    print(f"Remaining self.inv_freq references: {remaining}")

with open(TARGET, 'w') as f:
    f.write(content)

print(f"Successfully patched: {TARGET}")
