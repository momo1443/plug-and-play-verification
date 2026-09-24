# Agentic RLVR: Certificate-Grounded Reinforcement Learning for Verifiable Agents

> Train tool-using language models to emit explicit evidence artifacts and replayable certificates, so that correctness can be checked by deterministic programs instead of being deferred to test-time review.

![Agent-R1 training framework](image/framework.png)

**Figure 1.** Agentic RLVR builds on the Agent-R1 training framework. The policy alternates between reasoning and tool use, receives environment feedback, and is optimized from step-structured trajectories rather than flattened conversations.

## Overview

Agentic RLVR studies whether post-training can move verification from a bolted-on test-time procedure into the policy itself. A trained agent should not only produce a plausible final answer; every load-bearing action and answer claim should be accompanied by a structured certificate that a cheap deterministic verifier can inspect or replay.

The central training arm, **A9**, combines task success with domain-specific certificate quality:

```text
R_A9(tau) = w * R_task(tau) + (1 - w) * R_cert(tau),  w ~ Uniform(0, 1)
```

The first 50 optimizer steps use terminal task reward only for cold-start stabilization. After warmup, one mixing weight is sampled per prompt group and shared by all GRPO rollouts for that prompt. This preserves within-group comparability while exposing the policy to different task/certificate trade-offs during training.

The current implementation covers four domains:

1. Mathematical reasoning on DeepScaleR.
2. Multi-hop evidence retrieval on HotpotQA.
3. Code generation and execution on TACO.
4. Visual reasoning on Vision-R1.

All A9 process signals are deterministic and self-computable inside the training environment. Gold-derived or LLM-judged signals are retained as baselines or evaluation tools rather than being required by the A9 reward loop.

## Why Certificates?

Outcome-only RLVR can teach a model which final answers are rewarded without teaching it to expose why those answers should be trusted. Free-form chain-of-thought is also an unsuitable verification target: it may be incomplete, unfaithful, or expensive to evaluate.

Agentic RLVR externalizes the load-bearing part of the computation as an artifact:

| Property | Required behavior |
|---|---|
| Schema validity | Tool calls and certificates follow an explicit machine-readable contract. |
| Exact provenance | Evidence is tied to the environment artifact from which it was obtained. |
| Deterministic replay | A verifier can reproduce the declared computation or execution. |
| Answer coupling | The final answer is supported by the emitted evidence, not merely accompanied by it. |
| Fail-closed semantics | Missing, malformed, stale, or unverifiable artifacts receive no certificate credit. |
| Transfer | The learned behavior follows schemas supplied in context instead of memorizing a tool name. |

The long-term target is **certificate sufficiency**: a deterministic re-executor should be able to recover the final answer from the emitted evidence artifacts and declared computation alone.

## Method Comparison

| Method | Training signal | Process supervision | Deterministic | Replayable | External judge in reward loop |
|---|---|---:|---:|---:|---:|
| Terminal GRPO | Final-answer task verifier | No | Yes | Task-dependent | No |
| LLM-Judge GRPO | Frozen Qwen3.5-9B score | Semantic | No | No | Yes |
| A9 Agentic RLVR | Task reward + certificate verifier | Structured | Yes | Yes | No |

The frozen 9B judge arm is an important comparison: it tests whether a stronger semantic evaluator can replace hand-specified certificate checks. It is not part of the deterministic A9 mechanism.

## Certificate-Grounded GRPO

For a prompt `x`, GRPO samples a group of trajectories from the current policy:

```text
tau_1, ..., tau_G ~ pi_theta(. | x)
```

Each trajectory receives a terminal task score and a certificate score:

```text
r_i = w_x * R_task(tau_i) + (1 - w_x) * R_cert(tau_i)
```

where `w_x` is sampled once for the prompt group after warmup. Group-relative advantages are computed from the mixed rewards:

```text
A_i = (r_i - mean(r_1, ..., r_G)) / (std(r_1, ..., r_G) + epsilon)
```

The policy uses the clipped GRPO objective with a small reference-policy KL penalty. No critic or learned reward model is required.

### Step-Causal Credit Assignment

![Step-level trajectory representation](image/step-level-mdp.png)

**Figure 2.** The implementation preserves agent-step boundaries. Credit is attached to the decisions that produced tool calls, evidence artifacts, and final submissions instead of treating a multi-turn interaction as an undifferentiated token stream.

A trajectory is represented as a sequence of environment transitions:

```text
(state_t, model_action_t, tool_result_t, certificate_t, reward_t)
```

The training launchers enable `algorithm.grpo.credit_assignment=step_causal`, which aligns policy-gradient credit with the step-level MDP exposed by Agent-R1.

## Supported Domains

| Domain | Agent interaction | Certificate artifact | Deterministic check | A9 launcher |
|---|---|---|---|---|
| DeepScaleR | Reason, call calculator-style tools, finish | Parsed equations and declared numeric results | LaTeX-to-expression conversion, numerical equality, final-answer exact match | `examples/deepmath/run_deepscaler_tool_a9.sh` |
| HotpotQA | Search, inspect evidence, finish | Search/finish records with evidence provenance | Schema audit, document provenance, evidence-answer coupling | `examples/hotpotqa/run_a9_uniform.sh` |
| TACO | Write, execute, revise, submit code | Code hash, execution records, final submission certificate | Sandboxed replay against developer and private tests | `examples/taco/run_a9_uniform.sh` |
| Vision-R1 | Inspect image regions, crop, reason, finish | Crop coordinates, image-derived artifacts, answer link | Deterministic artifact construction and visual-certificate audit | `examples/vision_r1/run_a9_uniform.sh` |

### DeepScaleR: Equation Certificates

The math agent emits intermediate equations through a tool interface. The verifier parses supported mathematical expressions, evaluates both sides, and assigns process credit only to equations that are numerically valid. Terminal reward remains strict final-answer exact match.

```text
reason -> calculator/equation action -> verified result -> ... -> finish(answer)
```

This separates a correct final guess from a trajectory whose load-bearing calculations can be checked.

### HotpotQA: Provenance Certificates

The retrieval agent searches a fixed local corpus and must carry evidence provenance into its final response. Certificate checks cover the action schema, source identity, evidence validity, and coupling between retrieved support and the submitted answer.

```text
search(query) -> document evidence -> search(query) -> evidence -> finish(answer, citations)
```

Because the corpus is snapshot-able, certificate verification is local and deterministic.

### TACO: Execution Certificates

The code agent iterates through write, test, revision, and submission steps. The final certificate binds the submitted source to an artifact hash and replayable execution records. Untrusted programs run through a `bubblewrap` sandbox with time limits; model-visible data is separated from runner-only private tests.

```text
write(code) -> run(public tests) -> revise(code) -> run -> submit(code, execution certificate)
```

The reward contract fails closed when the code hash, execution record, sidecar lookup, or final submission is inconsistent.

### Vision-R1: Visual Evidence Certificates

The visual agent can request deterministic crops and use them as evidence for its answer. The certificate records image-derived artifacts and verifies that the declared visual evidence comes from the supplied image and is coupled to the final answer.

```text
inspect(image) -> crop(region) -> visual artifact -> ... -> finish(answer, evidence)
```

## Reward Arms

| Arm | Purpose | Reward definition |
|---|---|---|
| A0 / terminal | Outcome-only baseline | `R_task` |
| A1 | Strict task baseline for paired comparisons | Domain terminal verifier |
| A6 / judge | Frozen-Qwen3.5-9B baseline | Judge score with fail-closed parsing |
| A8-LR | Local reasoning ablation | Local step reward + terminal task reward |
| A9 cert-mix | Main certificate arm | Warmup: `R_task`; then `w R_task + (1-w) R_cert` |
| A9 format-strict | Formatting ablation | A9 reward gated by required completion protocol |

Reward-arm behavior is kept in domain-specific contracts and verifiers. Training code consumes the resulting scalar or step-structured scores without embedding domain logic in the optimizer.

## Frozen 9B Judge Baseline

The unified judge launcher trains a Qwen3.5-4B actor while a frozen Qwen3.5-9B model scores trajectories. The actor and judge use disjoint devices: seven GPUs for the actor and one GPU for the judge by default.

The judge returns a structured score from:

```text
{0.00, 0.25, 0.50, 0.75, 1.00}
```

DeepScaleR, TACO, and Vision-R1 use domain-specific structured rubrics. HotpotQA uses an evidence-coverage rubric. Invalid output, server failure, or schema mismatch fails closed instead of silently granting reward.

Run one domain:

```bash
bash examples/llm_judge/run_grpo_4b_judge_9b.sh deepscaler
bash examples/llm_judge/run_grpo_4b_judge_9b.sh hotpotqa
bash examples/llm_judge/run_grpo_4b_judge_9b.sh taco
bash examples/llm_judge/run_grpo_4b_judge_9b.sh vision
```

Run all four domains sequentially:

```bash
bash examples/llm_judge/run_grpo_4b_judge_9b.sh all
```

Inspect the resolved configuration without launching training:

```bash
AGENT_R1_JUDGE_CONFIG_ONLY=1 \
  bash examples/llm_judge/run_grpo_4b_judge_9b.sh deepscaler
```

## Training Pipeline

```text
dataset sample
    |
    v
domain prompt + tool schema
    |
    v
G policy rollouts per prompt
    |
    +--> tool/environment transitions
    |        |
    |        +--> evidence artifacts
    |        +--> replay records
    |
    v
terminal task verifier --------+
                               +--> A9 reward mixer
deterministic certificate -----+        |
                                        v
                              step-causal GRPO advantage
                                        |
                                        v
                              Qwen3.5 policy update
```

## Models and Training Framework

The primary actor is **Qwen3.5-4B**. Selected launchers also support **Qwen3.5-9B** configurations, while the LLM-judge baseline uses a frozen Qwen3.5-9B evaluator. Training and generation are implemented with **Agent-R1**, **veRL 0.7.0**, and **vLLM**.

Current launchers provide:

| Component | Configuration |
|---|---|
| Policy optimization | GRPO with step-causal credit assignment |
| Optional algorithm | PPO through the shared Agent-R1/veRL trainer |
| Actor optimization | FSDP, AdamW, gradient checkpointing |
| Generation | Asynchronous vLLM rollout workers |
| Group size | 4 rollouts per prompt |
| Actor learning rate | `1e-6` |
| Reference KL | `0.001`, `low_var_kl` |
| Sampling | temperature `1.0`, top-p `1.0`, top-k `-1` |
| Precision | bfloat16-compatible model/runtime configuration |
| Seed | `42` |

The paper protocol uses the first 10,000 training examples per domain. Because prompt and response lengths differ substantially across math, retrieval, code, and vision, batch size and total optimizer steps are domain-specific.

### Canonical 10K Budget

| Domain | Prompt batch | GRPO group | Optimizer steps | Prompt / completion limit |
|---|---:|---:|---:|---:|
| DeepScaleR | 20 | 4 | 500 | 4096 / 2048 tokens in ToolEnv |
| HotpotQA | 20 | 4 | 500 | 8192 / 1024 tokens |
| TACO | 20 | 4 | 500 | 8192 / 2048 tokens |
| Vision-R1 | 8 | 4 | 1250 | 8192 / 2048 tokens |

These are experiment-protocol settings used by the unified 4B-actor/9B-judge launcher. Individual A9 launchers retain configurable pilot defaults; override their environment variables when reproducing the 10K protocol.

## Project Structure

```text
Agent-R1/
|-- agent_r1/
|   |-- llm_agent/                 # agent rollout and tool integration
|   `-- trainer/                   # GRPO/PPO training entry points
|-- recipes/
|   |-- deepscaler/                # math tools, reward arms, equation verifier
|   |-- hotpotqa/                  # retrieval environment and provenance verifier
|   |-- taco_a9/                   # code protocol, sandbox, execution verifier
|   |-- vision_r1/                 # visual artifacts and certificate verifier
|   |-- mathvision/                # visual-math evaluation support
|   `-- reward_mixing.py           # shared A9 mixing utilities
|-- examples/
|   |-- deepmath/                  # DeepScaleR launchers
|   |-- hotpotqa/                  # HotpotQA launchers
|   |-- taco/                      # TACO launchers
|   |-- vision_r1/                 # Vision-R1 launchers
|   `-- llm_judge/                 # unified frozen-judge launcher
|-- scripts/                       # runtime compatibility patches
|-- tests/                         # unit and contract tests
`-- verl_patches/                  # rollout/trainer weight-sync support
```

Each certificate-enabled recipe follows the same separation of concerns:

```text
agent_flow.py       interaction loop and artifact assembly
prompts.py          task and tool-schema prompts
protocol.py / dsl.py
                    machine-readable action protocol
verifier.py         deterministic certificate validation
reward_contract.py  frozen reward semantics
reward_fn.py        veRL reward entry point
prepare_run.py      preflight checks and run manifest
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/momo1443/agentic-rlvr.git
cd agentic-rlvr
```

### 2. Prepare the Runtime

Follow the official [veRL installation guide](https://verl.readthedocs.io/en/latest/start/install.html) and use `verl==0.7.0`. The training recipes also require a compatible PyTorch, CUDA, vLLM, Ray, Transformers, and Flash Attention stack.

The launchers default to the following local model layout:

```text
../models/Qwen3.5-4B
../models/Qwen3.5-9B
```

Override model locations when needed:

```bash
export DEEPSCALER_TOOL_MODEL_PATH=/path/to/Qwen3.5-4B
export HOTPOTQA_MODEL_PATH=/path/to/Qwen3.5-4B
export TACO_A9_MODEL_PATH=/path/to/Qwen3.5-4B
export VISION_R1_MODEL_PATH=/path/to/Qwen3.5-4B
export AGENT_R1_JUDGE_MODEL=/path/to/Qwen3.5-9B
```

For Qwen3.5/FSDP2 compatibility in the pinned environment, inspect and apply the repository patches required by your installed package versions:

```bash
python scripts/patch_qwen35_lm_head_device.py
python scripts/patch_qwen35_rope_device.py
python scripts/patch_verl_fsdp2_ipc.py
```

The patch scripts validate their target code and should not be applied blindly to unrelated package versions.

### 3. Launch A9 Training

DeepScaleR ToolEnv on eight GPUs for a 500-step run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DEEPSCALER_TOOL_TRAIN_MAX_SAMPLES=10000 \
DEEPSCALER_TOOL_TOTAL_STEPS=500 \
bash examples/deepmath/run_deepscaler_tool_a9.sh
```

HotpotQA:

```bash
HOTPOTQA_TRAIN_MAX_SAMPLES=10000 \
HOTPOTQA_TOTAL_TRAINING_STEPS=500 \
bash examples/hotpotqa/run_a9_uniform.sh
```

TACO:

```bash
TACO_A9_TRAIN_MAX_SAMPLES=10000 \
TACO_A9_TOTAL_TRAINING_STEPS=500 \
bash examples/taco/run_a9_uniform.sh
```

TACO requires prepared model-safe parquet files, a runner-only sidecar and index, and `bubblewrap`. The launcher checks these artifacts before allocating the training job.

Vision-R1:

```bash
VISION_R1_TRAIN_MAX_SAMPLES=10000 \
VISION_R1_TOTAL_TRAINING_STEPS=1250 \
bash examples/vision_r1/run_a9_uniform.sh
```

Every main launcher validates required inputs before starting optimization. DeepScaleR, HotpotQA, and TACO additionally write a run manifest during preflight.

## Monitoring and Artifacts

Runs write into `../logs/<run-id>/` by default. A typical output directory contains:

```text
<run-id>/
|-- run_manifest.json              # recipes with prepare_run preflight
|-- train.log
|-- rollouts.jsonl
|-- validation/
`-- checkpoints/
```

Follow the console log:

```bash
tail -f ../logs/<run-id>/train.log
```

When TensorBoard logging is enabled by a launcher:

```bash
tensorboard --logdir ../logs/<run-id>/tensorboard --host 0.0.0.0 --port 6006
```

The rollout JSONL is part of the audit trail: it preserves tool interactions, reward components, and certificate metadata needed to inspect verifier behavior.

## Validation

Run the focused contract and reward tests before changing a domain verifier:

```bash
PYTHONPATH=. pytest -q tests/test_deepscaler_tool.py
PYTHONPATH=. pytest -q tests/test_hotpotqa_a9.py
PYTHONPATH=. pytest -q tests/test_taco_a9.py
PYTHONPATH=. pytest -q tests/test_vision_r1.py
PYTHONPATH=. pytest -q tests/test_cross_domain_llm_judge.py
```

Run the complete repository suite:

```bash
PYTHONPATH=. pytest -q tests
```

Evaluation should jointly report:

1. Final task quality.
2. Certificate/schema validity.
3. Exact provenance and deterministic replay success.
4. Evidence acquisition quality.
5. Transfer to held-out tasks and unseen tool schemas.
6. Legibility tax in accuracy, recall, latency, and token cost.
7. Verifier demotion at matched task quality.

## Design Principles

### Deterministic Training Signals

A9 uses only signals that can be recomputed from the frozen environment snapshot and the emitted trajectory. Judge-based rewards are isolated as experimental baselines.

### Frozen Reward Contracts

Each domain states exactly which artifacts earn credit. Changes to parsing, replay, or reward composition are covered by contract tests so that a training run cannot silently change scientific meaning.

### Artifact Separation

Private tests, gold-only metadata, and runner-side evidence remain outside model-visible prompts. Certificates reference the results of allowed tool interactions rather than leaking evaluator state.

### Fail-Closed Verification

Ambiguous parsing and incomplete evidence do not receive optimistic credit. This makes verifier failures visible and prevents malformed traces from exploiting permissive fallbacks.

### Honest Result Reporting

This repository provides training and evaluation infrastructure. Empirical claims should be added only with the exact checkpoint, dataset snapshot, reward contract, run manifest, and evaluation command used to obtain them.

## Infrastructure Patches

| Patch | Purpose |
|---|---|
| `verl_patches/bucketed_weight_transfer.py` | Bucketed rollout-to-trainer weight synchronization over ZMQ/IPC |
| `scripts/patch_qwen35_lm_head_device.py` | Qwen3.5 language-model-head placement under distributed training |
| `scripts/patch_qwen35_rope_device.py` | Qwen3.5 rotary-position device placement |
| `scripts/patch_verl_fsdp2_ipc.py` | veRL FSDP2 IPC integration for the pinned runtime |

## Acknowledgements

This project builds on [Agent-R1](https://github.com/AgentR1/Agent-R1), [veRL](https://github.com/volcengine/verl), [vLLM](https://github.com/vllm-project/vllm), [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), and [RAGEN](https://github.com/ZihanWang314/ragen).

## Citation

If you use the underlying Agent-R1 framework, please cite:

```bibtex
@misc{cheng2026agentr1unifiedmodularframework,
  title        = {Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning},
  author       = {Mingyue Cheng and Shuo Yu and Daoyu Wang and Qingchuan Li and Xiaoyu Tao and Jie Ouyang and Yucong Luo and Yitong Zhou and Qi Liu and Enhong Chen},
  year         = {2026},
  eprint       = {2511.14460},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2511.14460}
}
```

The citation entry for Agentic RLVR will be added when the corresponding paper is released.
