# Agentic RLVR

**Certificate-Grounded Agentic Reinforcement Learning for Verifiable Reasoning**

A research framework for training tool-using LLM agents to produce **verifiable, certificate-bearing reasoning** through end-to-end reinforcement learning. Built on [Agent-R1](https://github.com/AgentR1/Agent-R1) and [veRL](https://github.com/volcengine/verl).

---

## Research Goal

This repository implements the plug-and-play verifier interface and decoupled process/outcome reward composition in `37799_Plug_and_Play_Verifier_f.pdf`. Domain adapters extract artifacts and return a shared `VerificationResult`; reward composition and the optimizer consume bounded, step-attributed credits.

The checks establish local properties: numeric equalities, retrieved-text grounding and action coupling, developer-test lineage/replay, or crop replay and answer-claim coupling. They do not prove every reasoning step or reconstruct a globally correct answer from a certificate.

The implementation contract, model profiles, and validation limits are in [docs/paper-contract.md](docs/paper-contract.md).

---

## Core Contributions

### Certificate-Grounded Reward (A9)

The A9 reward arm composes **terminal task reward** (exact match) with a **deterministic process reward** that audits the agent's certificate trail:

- **Terminal reward**: Did the agent produce the correct answer?
- **Certificate process reward**: Did each reasoning step produce a verifiable certificate (e.g., numerically checkable equation, retrieved evidence citation)?

```
reward = w * terminal_EM + (1-w) * certificate_process_reward
```

A9 uses outcome-only reward for the first 50 updates. After warmup, one reproducible
uniform weight is shared by all rollouts of a prompt in the same update. A valid
final submission is required for composed reward; the answer may still be wrong
and the certificate may fail. Missing final submissions receive zero reward.
Validation always uses outcome-only reward. The historical `format_strict`
launcher now shares this eligibility rule.

### Deterministic Process Rewards

The proposed verifier supplies deterministic local checks. The frozen LLM judge is a separate comparison arm:

| Task | Process Signal | Verification Method |
|---|---|---|
| DeepScaler (math) | Per-step equation verification | LaTeX→Python conversion + numerical equality check |
| HotpotQA (multi-hop QA) | Certificate trail audit | Retrieved span/hash grounding + search/answer coupling |
| TACO (code) | Artifact and execution audit | Code lineage + developer-test replay |
| Vision-R1 (vision) | Visual artifact audit | Crop replay + answer-claim coupling |

### Step-Level Causal Advantage

Advantage estimation respects the **step-level MDP**: credit is assigned per agent step (tool call + observation), not per token or per trajectory. This aligns policy gradient signals with the actual decision boundaries that matter for agentic behavior.

---

## Tasks & Recipes

### Math Reasoning

| Recipe | Dataset | Reward | Launch |
|---|---|---|---|
| DeepScaler ToolEnv A1 | DeepScaleR-Preview-Dataset | Strict terminal EM | `scripts/extensions/deepscaler/run_deepscaler_tool_a1.sh` |
| DeepScaler ToolEnv A9 | DeepScaleR-Preview-Dataset | Uniform equation-process + terminal EM | `scripts/extensions/deepscaler/run_deepscaler_tool_a9.sh` |
| DeepScaler paper A1 | DeepScaleR-Preview-Dataset | Terminal EM | `scripts/deepscaler/run_deepscaler_a1.sh` |
| DeepScaler paper A9 | DeepScaleR-Preview-Dataset | Step-equation verification + EM | `scripts/deepscaler/run_deepscaler_a9_uniform.sh` |
| DeepMath | DeepMath-103K | `\boxed{}` terminal EM | `docs/archive/launchers/run_deepmath.sh` (historical; recipe absent) |
| AIME 2025 | AIME 2025 | EM | `scripts/deepscaler/run_aime2025_a0.sh` |

The ToolEnv answer-checking variants and DeepMath-103K recipe are extensions; paper math uses the shared single-turn flow with 2048/4096 tokens (4B) or 2048/5120 tokens (9B).

### Multi-Hop Question Answering

| Recipe | Dataset | Reward | Launch |
|---|---|---|---|
| A9 Certificate (HotpotQA) | HotpotQA | Certificate audit + EM | `scripts/hotpotqa/run_a9_uniform.sh` |
| A8-LR (Local Reasoning) | HotpotQA | Local reward + EM | `scripts/hotpotqa/run_rlvr.sh` |
| Validation | HotpotQA | EM only | `scripts/hotpotqa/run_validation.sh` |

### Supported Models

- Paper 4B: Qwen3.5-4B, full fine-tuning.
- Paper 9B: Qwen3.5-9B, LoRA rank/alpha 64 on seven projection modules.
- For renamed 9B checkpoints, set `AGENT_R1_MODEL_SCALE=9b`; explicit LoRA overrides are available.

---

## Architecture

Run entrypoints are grouped in [scripts/README.md](scripts/README.md). The
[repository guide](docs/README.md) separates paper code, optional extensions,
writing drafts, and historical notes.

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
git clone https://github.com/momo1443/plug-and-play-verification.git
cd plug-and-play-verification

# Apply patches (if needed for Qwen3.5 or FSDP2)
python scripts/patches/patch_qwen35_lm_head_device.py
python scripts/patches/patch_qwen35_rope_device.py
python scripts/patches/patch_verl_fsdp2_ipc.py
```

### Quick Start: DeepScaler with A9 Certificate Reward

```bash
# Place the prepared train.parquet and validation.parquet files under
# data/corpus/deepscaler/ before launching.

# Train with certificate-grounded reward
bash scripts/deepscaler/run_deepscaler_a9_uniform.sh
```

### Quick Start: HotpotQA with A9 Certificate Reward

```bash
# Prepare data (see recipe for details)
python -m recipes.hotpotqa.data_preprocess.process_hotpotqa --output_dir data/corpus/hotpotqa

# Train with certificate audit reward
bash scripts/hotpotqa/run_a9_uniform.sh
```

---

## Experimental Arms

| Arm | Description | Reward Composition |
|---|---|---|
| **A1 / GRPO** | Outcome only, matched domain interface | Terminal reward |
| **A9 / ours** | Deterministic verifier, group-shared mixture | Warmup: outcome; then `w E + (1-w) p` |
| **A6 / JUDGE** | Frozen Qwen3.5-9B process judge | Same mixture; no reference answer in judge input |
| **A2 / A3** | HotpotQA privileged gold-evidence ablations | Process only / fixed 0.5 mixture |
| **A0 / ReAct** | Untrained actor evaluation | No optimizer updates |

The formal launchers use a 10,000-row prepared training prefix, 500 updates,
4 rollouts/prompt, seed 42, actor LR 1e-6, and actor-loss KL 0.001. HotpotQA
uses the same certificate interface for every paper arm; TACO arms all use five
turns. A selected prefix of 10,000 rows does not mean every row is sampled in
500 updates: batch 8 consumes 4,000 prompts. Manifests record both quantities.

```bash
# Outcome-only / verifier: choose one command per experiment.
bash scripts/taco/run_a1_terminal.sh
bash scripts/taco/run_a9_uniform.sh
bash scripts/vision_r1/run_a1_terminal.sh
bash scripts/vision_r1/run_a9_uniform.sh
# Existing step-level PPO, using each domain's terminal-only interface.
bash scripts/ppo/run_terminal.sh deepscaler  # also hotpotqa, taco, vision
# Frozen judge: each domain uses its matched paper flow.
bash scripts/llm_judge/run_grpo_4b_judge_9b.sh deepscaler  # also hotpotqa, taco, vision
```

Task data, retrieval indexes, model files, and the Linux sandbox must be prepared
before a real run. In-training TACO/Vision validation is diagnostic; paper
results still require LiveCodeBench/MATH-Vision evaluation on the chosen checkpoint.

### Independent verification metrics

Every paper flow records all applicable deterministic checks for every reward
arm. `verification_consistency` implements `C_d`; `verified_success` implements
`E * C_d`. Earlier failures are retained after a repair. Missing evidence cannot
be inferred from positive credits alone. Code revision/retest counts are logged
separately and are not evidence of global reasoning correctness.

```bash
python -m agent_r1.evaluation.summarize validation.jsonl --output metrics.json
# A training dump must select one update, rather than pool checkpoints:
python -m agent_r1.evaluation.summarize rollouts.jsonl --global-step 500
```

AIME scoring selects one final answer before consulting the reference and
rejects fractional/out-of-range integers. Old AIME scores and dumps without
verification audits need re-evaluation; they are not retroactively corrected.

CPU regression checks (the flow tests mock generation and execution boundaries):

```bash
python -m pytest -q tests/test_paper_alignment.py tests/test_paper_launcher_config.py \
  tests/test_deepscaler_reward.py tests/test_verifier_reward.py tests/test_hotpotqa_a9.py \
  tests/test_taco_a9.py tests/test_cross_domain_llm_judge.py tests/test_process_judge_flows.py
```

Passing these tests verifies implementation contracts, not reproduction of the
paper's numerical results. Full GPU training and checkpoint benchmark runs are separate.

---

## Infrastructure Patches

| Patch | Purpose |
|---|---|
| `verl_patches/bucketed_weight_transfer.py` | ZMQ + IPC bucketed weight sync for faster rollout→trainer transfer |
| `scripts/patches/patch_qwen35_lm_head_device.py` | Fix Qwen3.5 lm_head device placement under FSDP |
| `scripts/patches/patch_qwen35_rope_device.py` | Fix Qwen3.5 RoPE device placement under FSDP |
| `scripts/patches/patch_verl_fsdp2_ipc.py` | Patch veRL FSDP2 IPC configuration |

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
