# Agentic RLVR

**Certificate-Grounded Agentic Reinforcement Learning for Verifiable Reasoning**

A research framework for training tool-using LLM agents to produce **verifiable, certificate-bearing reasoning** through end-to-end reinforcement learning. Built on [Agent-R1](https://github.com/AgentR1/Agent-R1) and [veRL](https://github.com/volcengine/verl).

---

## Research Goal

Can agentic post-training move verification from a bolted-on test-time procedure into the policy itself? We train tool-using models to make every load-bearing action and answer claim ship with an **explicit, structured certificate** that a cheap deterministic program can check, so certificate-complete behavior becomes the policy's default rather than an occasional result of prompting.

The learned behavior should be a **general schema-following and evidence-grounding capability**, including transfer to held-out or previously unseen tool schemas supplied in context — not memorization of one tool name or one benchmark format.

### Key Hypotheses

1. **Certificate sufficiency**: A deterministic re-executor can reproduce the final answer from emitted evidence artifacts and declared computation alone.
2. **Verifier demotion**: Post-training should reduce the need for test-time verification, revision, or rejection at matched quality.
3. **Legibility tax**: Any accuracy, recall, latency, or token-cost loss caused by making behavior checkable should be measured and minimized.

---

## Core Contributions

### Certificate-Grounded Reward (A9)

The A9 reward arm composes **terminal task reward** (exact match) with a **deterministic process reward** that audits the agent's certificate trail:

- **Terminal reward**: Did the agent produce the correct answer?
- **Certificate process reward**: Did each reasoning step produce a verifiable certificate (e.g., numerically checkable equation, retrieved evidence citation)?

```
reward = w * terminal_EM + (1-w) * certificate_process_reward
```

Two reward contracts are supported:
- **cert_mix** (fixed 0.4/0.6): Stable process-weight-dominant signal throughout training.
- **format_strict** (U(0,1) + format gate): Random per-trajectory weights with a hard gate that zeros all reward if the model fails to produce a valid finish tool call.

### Deterministic Process Rewards

All process rewards are **deterministic and execution-based** — no trainable reward model or LLM judge required:

| Task | Process Signal | Verification Method |
|---|---|---|
| DeepScaler (math) | Per-step equation verification | LaTeX→Python conversion + numerical equality check |
| DeepMath (math) | Format compliance (`\boxed{}`) | Regex extraction + symbolic equivalence |
| HotpotQA (multi-hop QA) | Certificate trail audit | Schema validation + evidence provenance + answer traceability |

### Step-Level Causal Advantage

Advantage estimation respects the **step-level MDP**: credit is assigned per agent step (tool call + observation), not per token or per trajectory. This aligns policy gradient signals with the actual decision boundaries that matter for agentic behavior.

---

## Tasks & Recipes

### Math Reasoning

| Recipe | Dataset | Reward | Launch |
|---|---|---|---|
| DeepScaler + A9 | DeepScaleR-1.5K | Step-equation verification + EM | `examples/deepmath/run_deepscaler_a9_uniform.sh` |
| DeepScaler (baseline) | DeepScaleR-1.5K | EM + format bonus | `examples/deepmath/run_deepscaler.sh` |
| DeepMath | DeepMath-103K | `\boxed{}` EM + format reward | `examples/deepmath/run_deepmath.sh` |
| AIME 2025 | AIME 2025 | EM | `examples/deepmath/run_aime2025_a0.sh` |

### Multi-Hop Question Answering

| Recipe | Dataset | Reward | Launch |
|---|---|---|---|
| A9 Certificate (HotpotQA) | HotpotQA | Certificate audit + EM | `examples/hotpotqa/run_a9_uniform.sh` |
| A8-LR (Local Reasoning) | HotpotQA | Local reward + EM | `examples/hotpotqa/run_rlvr.sh` |
| Validation | HotpotQA | EM only | `examples/hotpotqa_a9/run_validation.sh` |

### Supported Models

- Qwen3-4B / Qwen3.5-4B (primary)
- Qwen2.5-7B / Qwen2.5-9B
- LoRA and full fine-tuning supported

---

## Architecture

```
recipes/<task>/
  base.yaml                          # Hydra config
  data_preprocess/process_<task>.py  # Dataset preparation
  <task>_agent_flow.py               # Agent loop & certificate assembly
  reward_fn.py                        # Deterministic reward computation
  reward_contract.py                  # Frozen reward-arm semantics
  prompts.py                          # System/user/tool prompts
  protocol.py                         # Tool-call parsing
  verifier.py                         # Certificate verification
  env/                                # Environment services (optional)
```

Key design principles:
- **Algorithm-system decoupling**: Task workflows, rollout, rewards, and policy objectives evolve independently.
- **Step-level trajectory representation**: Each transition stores observation, action, environment feedback, reward, and termination — preserving action boundaries.
- **Flexible context management**: The environment decides what the model sees next; history can be appended, truncated, or augmented.

---

## Getting Started

### Environment Setup

Follow the [veRL installation guide](https://verl.readthedocs.io/en/latest/start/install.html). This project requires `verl==0.7.0`.

```bash
# Clone this repo
git clone https://github.com/momo1443/agentic-rlvr.git
cd agentic-rlvr

# Apply patches (if needed for Qwen3.5 or FSDP2)
python scripts/patch_qwen35_lm_head_device.py
python scripts/patch_qwen35_rope_device.py
python scripts/patch_verl_fsdp2_ipc.py
```

### Quick Start: DeepScaler with A9 Certificate Reward

```bash
# Prepare data
python -m recipes.deepscaler.data_preprocess.process_deepscaler --local_save_dir ~/data/deepscaler

# Train with certificate-grounded reward
bash examples/deepmath/run_deepscaler_a9_uniform.sh
```

### Quick Start: HotpotQA with A9 Certificate Reward

```bash
# Prepare data (see recipe for details)
python -m recipes.hotpotqa.data_preprocess.process_hotpotqa --local_save_dir ~/data/hotpotqa

# Train with certificate audit reward
bash examples/hotpotqa/run_a9_uniform.sh
```

---

## Experimental Arms

| Arm | Description | Reward Composition |
|---|---|---|
| **A9 cert_mix** | Certificate-grounded, fixed 0.4/0.6 split | 0.4 × EM + 0.6 × process |
| **A9 format_strict** | Certificate-grounded, random weights + format gate | U(0,1) × EM + (1-U) × process, gated |
| **A8-LR** | Local reasoning with per-step reward | Local reward + terminal EM |
| **A0** | Baseline: terminal EM only | EM |

---

## Infrastructure Patches

| Patch | Purpose |
|---|---|
| `verl_patches/bucketed_weight_transfer.py` | ZMQ + IPC bucketed weight sync for faster rollout→trainer transfer |
| `scripts/patch_qwen35_lm_head_device.py` | Fix Qwen3.5 lm_head device placement under FSDP |
| `scripts/patch_qwen35_rope_device.py` | Fix Qwen3.5 RoPE device placement under FSDP |
| `scripts/patch_verl_fsdp2_ipc.py` | Patch veRL FSDP2 IPC configuration |

---

## Monitoring

Training runs log to TensorBoard and `rollouts.jsonl`:

```bash
tensorboard --logdir <experiment_dir>/tensorboard --host 0.0.0.0 --port 6006
```

Validation-only runs produce evaluation metrics without training curves or rollout files.

---

## Acknowledgements

This project is built on [Agent-R1](https://github.com/AgentR1/Agent-R1), [veRL](https://github.com/volcengine/verl), [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), and [RAGEN](https://github.com/ZihanWang314/ragen).

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{cheng2026agentr1unifiedmodularframework,
      title={Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning},
      author={Mingyue Cheng and Shuo Yu and Daoyu Wang and Qingchuan Li and Xiaoyu Tao and Jie Ouyang and Yucong Luo and Yitong Zhou and Qi Liu and Enhong Chen},
      year={2026},
      eprint={2511.14460},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.14460},
}
```
