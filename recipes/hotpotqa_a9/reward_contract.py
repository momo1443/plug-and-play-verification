"""Frozen reward-arm semantics for A9 certificate-grounded experiments.

A9 runs as a single arm: cert_mix (0.8 terminal EM + 0.2 certificate process).
Protocol-null and cert-only sub-arms have been removed; the EM warmup phase
(steps 1–100) serves as the built-in protocol-tax control by comparing
against the A1 terminal-only baseline.
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


CONTRACT_CERT_MIX = CertificateRewardContract(terminal_weight=0.8, process_weight=0.2)

A9_CONTRACT_VERSION = "hotpotqa-a9-certificate-v1"

# Import-time invariant check
if abs(CONTRACT_CERT_MIX.terminal_weight + CONTRACT_CERT_MIX.process_weight - 1.0) > 1e-12:
    raise RuntimeError("A9 cert-mix reward weights must sum to one")
if CONTRACT_CERT_MIX.reward_horizon != 3:
    raise RuntimeError("A9 reward horizon must remain H=3")
