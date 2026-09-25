# Plug-and-Play Verifier for Decoupling Process Reward in Stabilizing Long-Horizon Agentic Reasoning

> A unified verifier interface that turns domain-specific checks of intermediate artifacts into bounded, step-attributed process rewards, without coupling verification logic to reward composition or policy optimization.

**ICLR 2027 submission (under review)** | **Agent-R1 + veRL** | **GRPO and PPO**

![Plug-and-play verifier framework](image/plug-and-play-verifier.png)

**Figure 1. Overview of the plug-and-play verifier framework.** Intermediate artifacts are extracted at each turn and routed through a unified interface to reusable domain-specific verifiers. The returned process credits are combined with the final outcome reward for policy optimization.

## Overview

Long-horizon agents must preserve useful progress across multiple reasoning steps and tool interactions. Outcome-only reinforcement learning cannot distinguish a partially useful failed trajectory from an entirely unproductive one: both receive the same zero reward. Actor-critic methods can provide denser credit through a learned value function, but that value function predicts return rather than explicitly checking intermediate artifacts.

We introduce a **plug-and-play verifier framework** that constructs process supervision by checking intermediate artifacts against task conditions and recorded environment evidence. The framework separates three concerns:

1. **Process verification:** a domain module extracts and verifies intermediate artifacts.
2. **Reward composition:** a shared two-stage schedule combines process and outcome rewards.
3. **Policy optimization:** GRPO or PPO consumes the composed, step-attributed rewards without accessing verifier internals.

Every verifier returns the same bounded credit contract. A search verifier can therefore be replaced by a code, math, multimodal, or alternative learned verifier without changing downstream reward mixing or policy optimization.

## Highlights

- **Unified verification contract.** Every module returns bounded credits, source-step indices, and audit metadata through the same `VerificationResult` interface.
- **Explicit process supervision.** Intermediate evidence, executions, calculations, and image crops are checked against task conditions and environment records.
- **Decoupled reward composition.** Verifier implementation details do not leak into the shared reward mixer or trainer.
- **Group-shared reward mixing.** After an outcome-only warmup, one reproducible `w ~ U(0,1)` is sampled per prompt group and shared across its GRPO rollouts.
- **Step-causal credit assignment.** Credits are assigned back to the policy steps that produced the verified artifacts before reward-to-go and group-relative normalization.
- **No judge required by our method.** The main verifier path is deterministic and uses neither gold evidence nor an LLM judge to compute process rewards.
- **Cross-domain and cross-algorithm results.** The same interface supports QA, code, mathematics, and multimodal reasoning, and improves both GRPO and PPO on HotpotQA.

## Method

![Method overview](image/method-overview.png)

**Figure 2. From sparse outcomes to plug-and-play process verification.** Outcome-only rewards collapse productive and unproductive failures to the same score. Our framework extracts artifacts, applies domain-specific checks behind a common interface, mixes bounded process reward with outcome reward, and performs step-level credit assignment before optimization.

### Problem Setting

For domain `d`, an agent receives input `x ~ D_d` and produces a trajectory of policy actions and environment observations:

```math
\tau = (a_1, o_1, \ldots, a_{T-1}, o_{T-1}, a_T).
```

The final action contains the submitted answer. We define two trajectory-level components:

```math
E_d(x, \tau) \in [0,1], \qquad P_d(x, \tau) \in [0,1],
```

where `E_d` measures final task correctness and `P_d` aggregates verified intermediate progress. A local verification pass establishes only the property checked by that rule; it does not imply that the final answer is correct.

### Unified Verification Contract

A verifier module consists of an artifact extractor `C_d`, a local verifier `V_d`, and an aggregation rule `omega_d`:

```math
C_d(x, \tau) = \{c_j\}_{j=1}^{m}, \qquad
v_j = V_d(c_j; \xi_j) \in [0,1].
```

The module returns bounded, step-attributed credits:

```math
\Phi_d(x, \tau; \xi_d)
= \left(\{(s_j, q_j, \eta_j)\}_{j=1}^{m}, \eta_d\right),
\qquad q_j \ge 0, \quad \sum_j q_j \le 1.
```

- `s_j` is the policy step that receives the credit.
- `q_j` is the bounded credit value.
- `eta_j` and `eta_d` retain credit-level and trajectory-level audit metadata.

The implementation exposes this contract directly:

```python
VerificationResult(
    credits=(
        VerificationCredit(
            step_index=source_step,
            score=bounded_score,
            audit=verification_record,
        ),
    ),
    audit=trajectory_audit,
)
```

Shared code reads only the step indices and credit values:

```math
p_{d,t}(x, \tau) = \sum_{j:s_j=t} q_j,
\qquad
P_d(x, \tau) = \sum_{t=1}^{T} p_{d,t}(x, \tau) \in [0,1].
```

Parsing, duplicate handling, evidence access, local rules, and normalization remain internal to each verifier. Audit metadata is preserved for inspection but does not enter reward composition or policy optimization.

### Two-Stage Reward Schedule

We use 50 outcome-only warmup updates. After warmup, one coefficient is sampled for each optimizer update `k` and prompt group `g`, then shared by all trajectories in that group:

```math
w_{k,g} \sim \mathcal{U}(0,1).
```

For trajectory `i`, the training reward is:

```math
R_{k,i} =
\begin{cases}
E_i, & 1 \le k \le 50, \\
w_{k,g}E_i + (1-w_{k,g})P_i, & k > 50.
\end{cases}
```

The implementation derives the pseudorandom weight from a hash of `(global_step, prompt_group_key)`, making the schedule reproducible. Validation always uses `w = 1` and measures final outcome correctness only.

### Step-Causal Credit Assignment

Outcome reward is assigned to the final policy step, while process credits retain the source-step attribution chosen by the verifier:

```math
r_{k,i,t}
= w_{k,g}\mathbf{1}\{t=T_i\}E_i
+ (1-w_{k,g})p_{i,t}.
```

The trainer computes undiscounted reward-to-go and normalizes returns across rollouts from the same prompt that reach the same step index. All valid policy tokens generated at a step receive that step's advantage; prompt tokens and environment observations are excluded from the policy surrogate and KL regularizer.

Changing a verifier therefore changes the process signal, while the reward schedule, advantage computation, token masking, and clipped policy objective remain unchanged.

## Domain Verifiers

| Domain | Training data | Evaluation | Intermediate artifact | Local verification |
|---|---|---|---|---|
| Mathematical reasoning | DeepScaleR-Preview-Dataset | AIME 2025 | Equations and declared calculations | Arithmetic consistency and final-answer verification |
| Multi-hop QA | HotpotQA train | HotpotQA validation | Search results, evidence provenance, final claims | Evidence validity, provenance, and action/answer coupling |
| Code generation | TACO | LiveCodeBench v6 | Code versions and execution records | Artifact lineage, sandboxed execution, and replay |
| Multimodal reasoning | Vision-R1-RL | MATH-Vision | Image crops and answer-linked visual claims | Crop replay and string-level answer/claim coupling |

All domains use the same verification output type even though their artifact parsers, permitted evidence, checking rules, and aggregation procedures differ.

### Dataset Links

- [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [TACO](https://huggingface.co/datasets/BAAI/TACO)
- [LiveCodeBench](https://huggingface.co/datasets/livecodebench/code_generation_lite)
- [Vision-R1-RL](https://huggingface.co/datasets/Osilly/Vision-R1-rl)
- [MATH-Vision](https://huggingface.co/datasets/MathLLMs/MathVision)
- [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
- [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025)

## Main Results

We train separately in each domain using the first 10,000 examples from its training split. Evaluation reports final task accuracy only. Values in parentheses are absolute improvements over outcome-only GRPO at the same model scale.

| Model / Method | AIME 2025 | HotpotQA | LiveCodeBench | MATH-Vision | Average |
|---|---:|---:|---:|---:|---:|
| **Qwen3.5-4B** | | | | | |
| ReAct | 26.67 | 46.78 | 40.47 | 75.72 | 47.41 |
| GRPO | 33.33 | 57.73 | 47.96 | 79.80 | 54.71 |
| GRPO + LLM-as-a-judge | 36.67 | 58.28 | 48.72 | **82.01** | 56.42 |
| **Ours (verification)** | **40.00 (+6.67)** | **60.18 (+2.45)** | **49.57 (+1.61)** | 81.51 (+1.71) | **57.82 (+3.11)** |
| **Qwen3.5-9B** | | | | | |
| ReAct | 36.67 | 52.56 | 49.00 | 79.47 | 54.43 |
| GRPO | 43.33 | 63.20 | 56.59 | 83.19 | 61.58 |
| **Ours (verification)** | **46.67 (+3.34)** | **66.67 (+3.47)** | **58.48 (+1.89)** | **84.80 (+1.61)** | **64.16 (+2.58)** |

The verifier framework improves over outcome-only GRPO on all four benchmarks at both model scales. With Qwen3.5-4B, it also exceeds the 9B LLM-judge reward baseline on three of four benchmarks and on average, without requiring an LLM judge in the reward loop.

> These numbers are transcribed from the current ICLR 2027 manuscript draft and should be reported together with the corresponding dataset snapshot, checkpoint, and evaluation configuration.

## Ablations

### Verification Mechanisms on HotpotQA

| Process feedback added to outcome-only GRPO | Accuracy (%) |
|---|---:|
| Outcome-only GRPO | 57.73 |
| + Execution validity | 57.18 |
| + LLM-as-a-judge | 58.28 |
| + Our verification | **60.18** |
| + Gold evidence | 61.40 |

Our verifier is within 1.22 points of gold-evidence matching while requiring neither annotated gold evidence nor an LLM judge at training time.

### Reward Composition on HotpotQA

| Configuration | Accuracy (%) |
|---|---:|
| ReAct | 46.78 |
| Outcome only (GRPO) | 57.73 |
| Process only | 47.27 |
| Outcome + process | **61.40** |

Process supervision is most effective as a complement to the final task objective, not as a replacement for it.

### GRPO and PPO

| Algorithm | Outcome only | + Verification | Gain |
|---|---:|---:|---:|
| GRPO | 57.73 | 60.18 | +2.45 |
| PPO | 55.10 | 57.60 | +2.50 |

The same HotpotQA verifier and reward schedule benefit both algorithms, supporting the intended decoupling between verification and policy optimization.

## Models and Training Framework

Experiments use **Qwen3.5-4B** and **Qwen3.5-9B**. The 9B policy is trained with LoRA rank `64` and scaling factor `alpha=64`. The implementation builds on **Agent-R1** and uses **veRL** for reward computation, training, and generation.

Within a domain and model scale, all RL methods share the same initialization, task interface, tools, fine-tuning method, data, and training budget.

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | `1e-6` |
| Training steps | 500 |
| LR schedule | Constant; no warmup |
| Global prompt batch size | 20 |
| GRPO group size | 4 rollouts |
| Mini-batch size | 20 prompts |
| Optimization epochs per rollout | 1 |
| Sampling | temperature `1.0`, top-p `1.0`, top-k `-1` |
| Weight decay | `0.01` |
| Gradient clipping | `1.0` |
| KL coefficient | `0.001` |
| Precision | bfloat16 |
| Random seed | 42 |

Domain-specific context limits:

| Model / Domain | Max prompt | Max completion | Agent turns |
|---|---:|---:|---:|
| Qwen3.5-4B / DeepScaleR | 2048 | 4096 | - |
| Qwen3.5-4B / HotpotQA | 8192 | 1024 | 4 |
| Qwen3.5-9B LoRA / DeepScaleR | 2048 | 5120 | - |

The manuscript experiments run on one Kubernetes node with **8 x NVIDIA A40 48GB GPUs**, using PyTorch FSDP and vLLM server-mode generation. Reported wall-clock time is approximately 37 hours for 500 training steps at 4B and 80 hours at 9B; evaluation inference takes approximately 15 minutes.

## Repository Structure

```text
Agent-R1/
|-- agent_r1/
|   |-- verifier/                  # unified credit contract and reward composition
|   |-- llm_agent/                 # agent rollout and tool integration
|   `-- trainer/                   # GRPO/PPO and step-causal advantage computation
|-- recipes/
|   |-- deepscaler/                # math artifacts and arithmetic verifier
|   |-- hotpotqa_a9/               # evidence artifacts and QA verifier
|   |-- taco_a9/                   # code lineage, sandbox, and execution verifier
|   |-- vision_r1/                 # crop artifacts and multimodal verifier
|   `-- llm_judge/                 # frozen-judge comparison arm
|-- examples/
|   |-- deepmath/                  # DeepScaleR launchers
|   |-- hotpotqa/                  # HotpotQA launchers
|   |-- taco/                      # TACO launchers
|   |-- vision_r1/                 # Vision-R1 launchers
|   `-- llm_judge/                 # unified 4B actor / 9B judge launcher
|-- tests/                         # verifier, reward, launcher, and trainer tests
|-- scripts/                       # runtime compatibility patches
`-- verl_patches/                  # rollout-to-trainer synchronization support
```

The shared verifier API is implemented in `agent_r1/verifier/reward.py`. Domain recipes own artifact extraction and local verification; the trainer owns reward-to-go, normalization, and policy optimization.

## Getting Started

### Installation

```bash
git clone https://github.com/momo1443/agentic-rlvr.git
cd agentic-rlvr
```

Follow the official [veRL installation guide](https://verl.readthedocs.io/en/latest/start/install.html) and use `verl==0.7.0`. The runtime also requires compatible versions of PyTorch, CUDA, vLLM, Ray, Transformers, and Flash Attention.

The launchers default to models stored beside the repository:

```text
../models/Qwen3.5-4B
../models/Qwen3.5-9B
```

### DeepScaleR: 8 GPUs, 10K Examples, 500 Steps

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DEEPSCALER_TOOL_TRAIN_MAX_SAMPLES=10000 \
DEEPSCALER_TOOL_TOTAL_STEPS=500 \
DEEPSCALER_TOOL_MAX_PROMPT_LENGTH=2048 \
DEEPSCALER_TOOL_MAX_RESPONSE_LENGTH=4096 \
bash examples/deepmath/run_deepscaler_tool_a9.sh
```

### HotpotQA

```bash
HOTPOTQA_TRAIN_MAX_SAMPLES=10000 \
HOTPOTQA_TOTAL_TRAINING_STEPS=500 \
bash examples/hotpotqa/run_a9_uniform.sh
```

### TACO

```bash
TACO_A9_TRAIN_MAX_SAMPLES=10000 \
TACO_A9_TOTAL_TRAINING_STEPS=500 \
bash examples/taco/run_a9_uniform.sh
```

TACO requires prepared model-safe parquet files, a runner-only private-test sidecar and index, and `bubblewrap`. The launcher fails before training if any required artifact is missing.

### Vision-R1

```bash
VISION_R1_TRAIN_MAX_SAMPLES=10000 \
VISION_R1_TOTAL_TRAINING_STEPS=1250 \
bash examples/vision_r1/run_a9_uniform.sh
```

Vision-R1 uses a prompt batch of 8, so 10,000 examples correspond to 1,250 optimizer steps.

## LLM-as-a-Judge Baseline

The comparison arm trains a Qwen3.5-4B actor while a frozen Qwen3.5-9B model scores process quality. By default, GPUs 0-6 serve the actor and GPU 7 serves the judge.

The judge supplies **process rewards**, while deterministic binary final-answer correctness remains the outcome reward. The implementation contract is documented in [docs/experiments/llm-process-judge.md](docs/experiments/llm-process-judge.md). Math/code/vision now use causal intermediate-step scoring (`qwen35-9b-causal-process-v2`), replacing the earlier terminal-judge baseline. Results above must not be attributed to this revised baseline without new training/evaluation runs.


```bash
bash examples/llm_judge/run_grpo_4b_judge_9b.sh deepscaler
bash examples/llm_judge/run_grpo_4b_judge_9b.sh hotpotqa
bash examples/llm_judge/run_grpo_4b_judge_9b.sh taco
bash examples/llm_judge/run_grpo_4b_judge_9b.sh vision
```

Run all four jobs sequentially:

```bash
bash examples/llm_judge/run_grpo_4b_judge_9b.sh all
```

Inspect the resolved device and model configuration without launching:

```bash
AGENT_R1_JUDGE_CONFIG_ONLY=1 \
bash examples/llm_judge/run_grpo_4b_judge_9b.sh deepscaler
```

## Monitoring and Audit Artifacts

Training outputs are written to `../logs/<run-id>/` by default:

```text
<run-id>/
|-- run_manifest.json              # recipes with prepare_run preflight
|-- train.log
|-- rollouts.jsonl
|-- validation/
`-- checkpoints/
```

```bash
tail -f ../logs/<run-id>/train.log
```

`rollouts.jsonl` retains tool interactions, component rewards, verifier credits, source-step attribution, and audit metadata. This record is the primary artifact for checking whether a verifier rewarded the intended behavior.

## Validation

Run verifier and reward-contract tests:

```bash
PYTHONPATH=. pytest -q tests/test_verifier_reward.py
PYTHONPATH=. pytest -q tests/test_deepscaler_tool.py
PYTHONPATH=. pytest -q tests/test_hotpotqa_a9.py
PYTHONPATH=. pytest -q tests/test_taco_a9.py
PYTHONPATH=. pytest -q tests/test_vision_r1.py
PYTHONPATH=. pytest -q tests/test_cross_domain_llm_judge.py
PYTHONPATH=. pytest -q tests/test_step_grpo.py
```

Run the complete suite:

```bash
PYTHONPATH=. pytest -q tests
```

## Limitations

- Current deterministic verifiers establish local properties such as evidence grounding, arithmetic consistency, artifact lineage, and execution replay; they do not establish the full semantic correctness of every intermediate step.
- Each domain is trained separately, and current interaction horizons remain shorter than open-ended web or software-engineering tasks.
- Experiments cover models up to 9B, with the 9B policy trained using LoRA.
- Learned verifiers and calibrated probabilistic decision models can be added behind the same bounded-credit contract, but are left to future work.

## Acknowledgements

This implementation builds on [Agent-R1](https://github.com/AgentR1/Agent-R1), [veRL](https://github.com/volcengine/verl), and [vLLM](https://github.com/vllm-project/vllm).

## Citation

The paper is currently under double-blind review. A final citation will be added after review. For anonymous artifact references, use:

```bibtex
@article{anonymous2027plugandplayverifier,
  title   = {Plug-and-Play Verifier for Decoupling Process Reward in Stabilizing Long-Horizon Agentic Reasoning},
  author  = {Anonymous Authors},
  journal = {Under review at ICLR},
  year    = {2027}
}
```

For the underlying training framework, please also cite:

```bibtex
@misc{cheng2025agentr1,
  title         = {Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning},
  author        = {Mingyue Cheng and Shuo Yu and Daoyu Wang and Qingchuan Li and Xiaoyu Tao and Jie Ouyang and Yucong Luo and Yitong Zhou and Qi Liu and Enhong Chen},
  year          = {2025},
  eprint        = {2511.14460},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
