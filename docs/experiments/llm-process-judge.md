# Frozen Qwen3.5-9B process-reward baseline

The policy defaults to Qwen3.5-4B and the inference-only judge to Qwen3.5-9B. The judge supplies process rewards; it never replaces the deterministic binary outcome reward. This changes the earlier math/code/vision terminal-judge implementation. Historical scores require new training and evaluation before being reported for this version.

## Domain contracts

| Domain | Judge input and process reward | Binary outcome |
| --- | --- | --- |
| HotpotQA | Existing A6: question, search queries, accumulated retrieved passages and reference supporting facts. Judge identifies supported fact IDs; each search receives newly covered facts divided by the total reference facts. | Normalized final-answer exact match. |
| Math / DeepScaleR | Each non-terminal reasoning/tool action and observed feedback, with earlier steps as context. | Existing strict final-answer math matcher. |
| Code / TACO | Intermediate code writes/revisions and developer-test actions with observed feedback. No private tests or private test outcomes enter the judge prompt. | One iff the submitted program passes all private tests. |
| Vision-R1 | Intermediate inspection action, its purpose and feedback, with original image and current crop. | Existing final-answer normalization/math-equivalence check. |

Math/code/vision use `qwen35-9b-causal-process-v2`. The judge returns one of `{0, 0.25, 0.5, 0.75, 1}` per intermediate agent turn. It evaluates correct, relevant, non-redundant progress, not final-answer correctness. There is no reference final answer or final submission in these prompts. Judging runs after rollout, using a separate causal prefix for each step; future actions/observations are excluded.

For configured maximum agent turns `T_max`, a step receives `raw_score / T_max`. These credits sum to at most one, including when a rollout exhausts its budget without submitting. Invalid actions receive zero without a judge call. A trajectory that immediately submits has no process credit. This is **agent-turn-level** scoring, not a separate judgment of every sentence within a turn.

The shared `VerificationResult` carries scores and source-step indices. Each domain's existing reward composer assigns `w * binary_outcome` to the final transition and `(1-w) * process_credit[t]` to its source transition. Math/code/vision retain 50 outcome-only warmup updates by default, followed by their existing prompt-group-shared uniform mixing schedule. These three domains skip process-judge calls during validation, using only binary outcome rewards. HotpotQA retains its existing A6 fixed 0.5 outcome / 0.5 process mixture and supporting-fact supervision; it is not a matched schedule or supervision setting for all other arms. Its validation still logs judge evidence coverage, but that coverage contributes no validation reward.

Wrong final answers and missing submissions yield binary outcome zero, while useful earlier process credit can remain nonzero. A judge transport/parse failure sets `judge_invalid`, which the existing trainer gate checks before any optimizer update. Such a failure is not treated as an ordinary zero-quality action.

## Run

From the repository root, on the existing GPU training host with prepared datasets and model files:

```bash
bash scripts/llm_judge/run_grpo_4b_judge_9b.sh deepscaler
bash scripts/llm_judge/run_grpo_4b_judge_9b.sh hotpotqa
bash scripts/llm_judge/run_grpo_4b_judge_9b.sh taco
bash scripts/llm_judge/run_grpo_4b_judge_9b.sh vision
```

The `all` argument runs these sequentially. Defaults allocate GPUs 0-6 to the actor and GPU 7 to the frozen judge. `AGENT_R1_JUDGE_MODEL` can override the model path, so retain the actual model identity in run records. Vision requires the Qwen3.5 multimodal checkpoint and a serving configuration that accepts the two image inputs; this must be smoke-tested on the training host.

Configuration-only inspection does not load a model or launch training:

```bash
AGENT_R1_JUDGE_CONFIG_ONLY=1 bash scripts/llm_judge/run_grpo_4b_judge_9b.sh all
```

## Verification scope

Focused tests cover causal judge prompts, bounded/source-attributed rewards, distinct binary outcomes, incomplete trajectories, warmup/validation, fail-closed metadata, and image-aware local/remote request caching. `test_process_judge_flows.py` executes production flow methods with mocked generation, environment and outcome-matcher boundaries, omitting heavy runtime imports through AST loading. These CPU tests do not validate Ray/veRL integration, GPU serving, learned judge quality, or training performance.

```bash
PYTHONPATH=.:tests python -m unittest test_cross_domain_llm_judge test_process_judge_flows test_hotpotqa_judge_server test_hotpotqa_reward_arms
```
