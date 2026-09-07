"""Frozen reward-arm semantics for A9 certificate-grounded experiments.

A9 supports two reward composition modes:

1. cert_mix (fixed 0.4/0.6): The fixed 0.4/0.6 split replaces the previous
   uniform U(0,1) sampling.  The higher process weight ensures the certificate
   signal remains dominant throughout training.

2. format_strict (uniform U(0,1) + format gate): Per-trajectory random weights
   w ~ U(0,1) combine terminal EM and certificate process reward.  A format
   gate zeros ALL reward when the model fails to produce a valid finish tool
   call, forcing protocol compliance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CertificateRewardContract:
    """Optimizer-visible reward weights for A9 cert-mix."""

    terminal_weight: float
    process_weight: float
    reward_horizon: int = 3
    final_response_mask: int = 1
    format_gate: bool = False


CONTRACT_CERT_MIX = CertificateRewardContract(terminal_weight=0.4, process_weight=0.6)
CONTRACT_FORMAT_STRICT = CertificateRewardContract(
    terminal_weight=0.5, process_weight=0.5, format_gate=True,
)

A9_CONTRACT_VERSION = "hotpotqa-a9-certificate-v1"

# Import-time invariant check
if abs(CONTRACT_CERT_MIX.terminal_weight + CONTRACT_CERT_MIX.process_weight - 1.0) > 1e-12:
    raise RuntimeError("A9 cert-mix reward weights must sum to one")
if CONTRACT_CERT_MIX.reward_horizon != 3:
    raise RuntimeError("A9 reward horizon must remain H=3")
