# Plug-and-Play Verifier for Decoupling Process Reward in Stabilizing Long-Horizon Agentic Reasoning

**PNP adds verified process rewards to multi-turn reinforcement learning.** A shared interface connects domain verifiers to GRPO and PPO across mathematics, QA, code, and vision.

[Results](#results) · [Quick Start](#quick-start) · [Method](#method) · [Evaluation](#evaluation) · [Documentation](#documentation)

Built on [Agent-R1](https://github.com/AgentR1/Agent-R1), [veRL](https://github.com/volcengine/verl), and [vLLM](https://github.com/vllm-project/vllm).

![Overview of the plug-and-play verifier](assets/images/plug-and-play-verifier.png)

## Overview

PNP checks numeric equalities, retrieved evidence, code executions, and image crops. Verified work receives credit at its source step, even when the final answer is incorrect or missing. The proposed verifiers use deterministic checks; frozen LLM judges and gold evidence are comparison arms.

| Domain | Training dataset | Reported evaluation benchmark |
| --- | --- | --- |
| Mathematics | DeepScaleR-Preview-Dataset | AIME 2025 |
| Multi-hop QA | HotpotQA | HotpotQA validation |
| Code | TACO | LiveCodeBench |
| Vision | Vision-R1-RL | MATH-Vision |

## Results

Manuscript Tables 1–3 report the following scores (%). `Avg.` is the unweighted mean across four benchmarks; AIME 2025 uses at most five policy generations.

### Qwen3.5-4B

| Method | AIME 2025 | HotpotQA | LiveCodeBench | MATH-Vision | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| ReAct | 26.67 | 46.78 | 40.47 | 75.72 | 47.41 |
| GRPO | 33.33 | 57.73 | 47.96 | 79.80 | 54.71 |
| GRPO + LLM-as-a-judge | 36.67 | 58.28 | 48.72 | **82.01** | 56.42 |
| **Ours (verification)** | **40.00** | **60.18** | **49.57** | 81.51 | **57.82** |

### Qwen3.5-9B

| Method | AIME 2025 | HotpotQA | LiveCodeBench | MATH-Vision | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| ReAct | 36.67 | 52.56 | 49.00 | 79.47 | 54.43 |
| GRPO | 43.33 | 63.20 | 56.59 | 83.19 | 61.58 |
| **Ours (verification)** | **46.67** | **66.67** | **58.48** | **84.80** | **64.16** |

Verification improves average accuracy over outcome-only GRPO by **3.11 percentage points at 4B** and **2.58 at 9B**.

### Training dynamics

![Training accuracy of Qwen3.5-4B with verification and an LLM process judge](assets/images/training-accuracy.png)

### Reuse across policy optimizers

HotpotQA, Qwen3.5-4B (Table 3):

| Algorithm | Outcome only | + Verification | Gain (percentage points) |
| --- | ---: | ---: | ---: |
| GRPO | 57.73 | **60.18** | +2.45 |
| PPO | 55.10 | **57.60** | +2.50 |

<details>
<summary><strong>HotpotQA ablations: verification source and reward composition</strong></summary>

Qwen3.5-4B with GRPO (Table 2):

| Process feedback added to outcome-only GRPO | Accuracy (%) |
| --- | ---: |
| None: outcome-only GRPO | 57.73 |
| Frozen LLM judge | 58.28 |
| **Our verifier** | **60.18** |
| Gold evidence (privileged upper bound) | 61.40 |

Reward composition with **privileged gold-evidence process rewards**:

| Reward configuration | Accuracy (%) |
| --- | ---: |
| ReAct | 46.78 |
| Outcome only | 57.73 |
| Gold process only | 47.27 |
| Outcome + gold process | 61.40 |

</details>

## Quick Start

### 1. Set up the environment

Use Linux with NVIDIA GPUs. These commands install the pinned Python 3.10 environment for Qwen3.5 mathematics; other domains require their task-specific dependencies.

```bash
git clone https://github.com/momo1443/plug-and-play-verification.git
cd plug-and-play-verification

conda create -n pnp python=3.10 -y
conda activate pnp
python -m pip install uv
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
uv pip install --python "$PYTHON_BIN" \
  --override recipes/deepscaler/requirements-qwen35-overrides.txt \
  -r recipes/deepscaler/requirements-qwen35.txt
```

Runtime compatibility patches are documented in the [mathematics setup guide](recipes/deepscaler/README.md#local-qwen35-setup).

### 2. Prepare data and model weights

| Domain | Required inputs | Preparation guide |
| --- | --- | --- |
| DeepScaleR | Source JSON and prepared splits | [Mathematics recipe](recipes/deepscaler/README.md) |
| HotpotQA | Splits, retrieval corpus/index, embedding model, evidence sidecar | [HotpotQA recipe](recipes/hotpotqa/README.md) |
| TACO | Splits, test sidecars, Linux `bubblewrap` | [Preparation](recipes/taco_a9/prepare_data.py) · [Sidecar index](recipes/taco_a9/build_sidecar_index.py) |
| Vision-R1 | Official train/test splits and images | [Preparation](recipes/vision_r1/prepare_data.py) |

For mathematics, prepare the local corpus with:

```bash
python -m recipes.deepscaler.prepare_data
```

Place weights in `models/Qwen3.5-4B` or `models/Qwen3.5-9B`, or pass `--model-path`. Data, weights, and run outputs are excluded from Git.

### 3. Train with verification

Formal profiles use **eight NVIDIA A40 GPUs and 500 training updates**:

```bash
bash run_pnp.sh deepscaler \
  --model Qwen3.5-4B \
  --model-path /path/to/Qwen3.5-4B \
  --gpus 0,1,2,3,4,5,6,7
```

The launcher also accepts `hotpotqa`, `taco`, `vision_r1`, and `--model Qwen3.5-9B`. Defaults select 10,000 training rows and four rollouts per prompt; 4B uses full fine-tuning and 9B uses LoRA (rank/alpha 64).

See the [launcher index](scripts/README.md) for comparison commands and the [paper contract](docs/paper-contract.md) for batch sizes and token budgets. [Frozen-judge runs](docs/experiments/llm-process-judge.md) allocate seven GPUs to the actor and one to the judge.

## Method

![The plug-and-play process verification method](assets/images/method-overview.png)

Domain adapters return source-step credits and audit records through [`VerificationResult`](agent_r1/verifier/reward.py). Credits are nonnegative and sum to at most one. Failed applicable checks remain in normalization; inapplicable checks are excluded.

For final outcome reward `E` and process credit `p_t`, the composer assigns:

```math
r_t=w_{k,g}\,\mathbf{1}\{t=T\}\,E+(1-w_{k,g})p_t.
```

Updates 1–50 use outcome only (`w=1`). Later updates sample one reproducible `w ~ Uniform(0,1)` per update and prompt group, shared by its rollouts. Missing submissions receive zero outcome reward while retaining verified process credits. GRPO and PPO consume the composed step rewards; policy loss applies only to generated tokens.

Local consistency `C_d=1` requires at least one applicable check and all applicable checks to pass, independently of final submission. Later corrections do not erase earlier failures. Reward and audit definitions are detailed in the [paper contract](docs/paper-contract.md).

<details>
<summary><strong>Example: verification on HotpotQA</strong></summary>

![A concrete HotpotQA verification example](assets/images/hotpotqa-verification-example.png)

</details>

## Evaluation

Accuracy scores final submissions over all selected questions; missing or invalid submissions score zero. Process consistency is recorded separately. The shared evaluator uses pinned, unmodified official HotpotQA, LiveCodeBench, and MATH-V graders; AIME uses exact integer equality.

Export FSDP checkpoints before inference and merge LoRA weights when applicable:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /path/to/global_step_500/actor \
  --target_dir /path/to/exported-checkpoint
```

Install the isolated grader environment, then check evaluation prerequisites:

```bash
bash scripts/evaluation/setup_official_graders.sh
bash run_pnp_eval.sh aime2025 \
  --model /path/to/exported-checkpoint \
  --dataset /path/to/aime2025/test.parquet \
  --dataset-version 2025 \
  --dataset-revision LOCAL_AIME2025_SNAPSHOT \
  --preflight-only
```

Replace the paths and snapshot identifier with your actual inputs; remove `--preflight-only` to run. The runner defaults to eight GPUs and saves manifests, outputs, and summaries under `logs/`. Use matching datasets, interfaces, graders, and budgets across policies. All four benchmark commands are in the [evaluation guide](docs/pnp_evaluation.md).

## Documentation

| Resource | Contents |
| --- | --- |
| [Repository guide](docs/README.md) | Code layout |
| [Launcher index](scripts/README.md) | Training and comparison commands |
| [Paper contract](docs/paper-contract.md) | Rewards, budgets, and metrics |
| [Shared settings](scripts/common/README.md) | Model and optimizer options |
| [Evaluation guide](docs/pnp_evaluation.md) | Benchmarks and official graders |
| [CPU checks](tests/README.md) | Test instructions |

Run CPU checks after installing the task and grader dependencies:

```bash
PYTHONPATH=.:tests python -m pytest -q tests
```

Reproducing manuscript scores requires the corresponding checkpoints and frozen data. CPU tests check implementation behavior.

## Acknowledgements and License

Built on the projects and datasets linked above. The project uses the [MIT License](LICENSE); vendored graders retain their [upstream licenses](third_party/official_graders/README.md).
