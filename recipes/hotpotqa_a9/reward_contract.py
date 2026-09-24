"""Frozen reward-arm semantics for A9 certificate-grounded experiments.

A9 uses one prompt-group-shared uniform mixture after outcome-only warmup:

  w ~ U(0, 1), reward = w * terminal_EM + (1-w) * certificate_process.

All rollouts for the same prompt in an optimizer update use the same w.

The strict format-gate launcher keeps the same uniform mixture, but adds a
separate protocol-compliance ablation that zeros rewards for missing finishes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CertificateRewardContract:
    """Optimizer-visible A9 reward schedule parameters."""

    terminal_weight: float
    process_weight: float
    reward_horizon: int = 3
    final_response_mask: int = 1
    format_gate: bool = False


CONTRACT_CERT_MIX = CertificateRewardContract(terminal_weight=0.5, process_weight=0.5)
CONTRACT_FORMAT_STRICT = CertificateRewardContract(
    terminal_weight=0.5, process_weight=0.5, format_gate=True,
)

A9_CONTRACT_VERSION = "hotpotqa-a9-certificate-group-uniform-v3"

# Import-time invariant check
if abs(CONTRACT_CERT_MIX.terminal_weight + CONTRACT_CERT_MIX.process_weight - 1.0) > 1e-12:
    raise RuntimeError("A9 expected uniform reward weights must sum to one")
if CONTRACT_CERT_MIX.reward_horizon != 3:
    raise RuntimeError("A9 reward horizon must remain H=3")
