# Paper implementation contract

Target: `2027_PnP_Verifier__arXiv_.pdf`, Sections 3–4 and Appendices B–C. The reported mathematics results use the protocol below with at most five policy generations.

Formal runs use one node with eight NVIDIA A40 48GB GPUs, matching Appendix C.
Both 4B and 9B verifier/outcome/PPO profiles use all eight actor GPUs. Frozen
Qwen3.5-9B process-judge comparisons use seven actor GPUs and one dedicated judge
GPU on the same eight-GPU node.

- Verifiers return nonnegative, bounded, source-step credits. Reward composition and policy optimization do not interpret domain artifacts.
- Outcome reward is placed at the last policy step. Process credits remain at their source steps. For the proposed verifier and judge arms, the first 50 updates use outcome only; subsequent updates share one reproducible uniform weight per prompt group. Validation uses outcome only.
- Terminal eligibility requires a final submission in the domain protocol, regardless of whether that answer is correct or its certificate passes. An incomplete trajectory receives zero outcome reward while retaining independently verified process credits under the same mixture schedule.
- GRPO uses undiscounted returns, prompt/step grouping, sample standard deviation, and the Appendix B singleton convention. Policy and KL losses use policy tokens only.
- Within each domain, outcome-only, verifier, and frozen-judge arms share prompts, tools, final-answer extraction, generation limits, and model adaptation.
- Formal runs select the first 10,000 rows of the prepared training split and use 500 updates, 4 rollouts/prompt, seed 42, learning rate 1e-6, and actor-loss KL coefficient 0.001. Per the updated hardware configuration, 4B uses prompt batch 20 and 9B uses batch 8. At 500 updates these sample 10,000 and 4,000 prompts respectively, with four rollouts each; the selected training prefix remains 10,000 rows. Manifests record selection size and actual sampled prompt count separately. These batch sizes match Appendix C.

| Domain | 4B batch | 9B batch | Prompt tokens | Completion tokens | Policy steps |
|---|---:|---:|---:|---:|---:|
| DeepScaleR | 20 | 8 | 2048 initial; 13312 history | 10240 total per trajectory (2048 per turn) | 5 |
| HotpotQA | 20 | 8 | 8192 | 1024 | 4 |
| TACO | 20 | 8 | 8192 | 2048 | 5 |
| Vision-R1 | 20 | 8 | 8192 | 2048 | 3 |

9B uses LoRA rank/alpha 64 and the seven projection modules listed in Appendix C; 4B uses full fine-tuning. DeepScaleR permits up to five reasoning generations by default (`DEEPSCALER_MAX_STEPS`), with a shared total token budget and per-turn cap `ceil(total / max_steps)`. A reference-independent extracted final answer ends the trajectory; incomplete trajectories retain their steps/masks and independently verified process credits, with zero outcome reward. Equation/judge verification occurs retrospectively and credits remain on their source turns. No correctness or verification feedback enters model context. Set `DEEPSCALER_MAX_STEPS=1` for an optional single-generation setting. Appendices B and C describe the five-generation mathematics protocol used for the reported results. The answer-checking ToolEnv remains a separate optional protocol.

The PNP process verifiers exclude inapplicable checks from reward normalization. Applicability follows the domain protocol, not whether a check passes or its evidence parses. Missing or malformed required evidence and failed applicable checks remain in the denominator; an empty applicable set yields process reward zero.

| Domain | Participating process-reward positions | Excluded positions |
|---|---|---|
| DeepScaleR | Supported extracted numerical equations, averaged within equation-bearing text segments and then policy turns, including failed checks | Text segments and turns with no supported extractable equation |
| HotpotQA | Actual search/finish certificate transitions within the reward horizon, including missing, malformed, or failed required certificates | Bootstrap search, unused horizon positions, and actions without an applicable certificate check |
| TACO | Tested code versions and the certificate check triggered by a submission attempt or provided certificate | Untested intermediate versions and an untriggered submission-certificate check |
| Vision-R1 | Required cited crops; with an unreadable certificate, recorded crops, declared missing evidence positions, and a failed certificate position if submission or certificate attempted | Root image, uncited crops when a readable certificate selects evidence, and an untriggered submission-certificate check |

For TACO and Vision-R1, attempted submissions with missing or malformed required certificates still retain the failed check. The exclusion rule does not remove actions, tokens, or trajectories, change outcome grading, or change outcome-only/frozen-judge control arms.

Evaluation extracts one final answer independently of the reference, then scores it. Equation (8) sets local consistency C_d to one exactly when at least one check is applicable and every applicable check passes. Final-submission eligibility is recorded separately and does not gate C_d; failures and missing required evidence remain applicable, and later repairs do not erase earlier failures. Equation (9) multiplies C_d by the final outcome reward. An incomplete but locally consistent trajectory can therefore have C_d=1 and zero accuracy/verified success. Audit metrics do not affect optimizer rewards. Revision and retesting are recorded separately from strict consistency.

Implementation map: `agent_r1/verifier` owns contracts and composition; domain recipes own checks and adapters; `agent_r1/trainer/ppo` owns advantages/objectives; `scripts/*` owns runnable profiles; `agent_r1/evaluation` and `scripts` own offline metrics. CPU checks cover answer extraction, contracts, real launcher arguments, and agent-flow wiring. GPU training and benchmark scores require separate runs and are not implied by passing tests.

HotpotQA A6 uses the same causal, gold-free judge protocol as the other three domains: only the question, policy actions, and retrieved passage text enter judge prompts. It uses 50 outcome-only warmup updates, then one prompt-group-shared w ~ Uniform(0,1) outcome/process mixture; validation remains outcome-only. Gold supporting facts and reference answers are excluded from judge input. A2 remains gold process-only, and A3 remains a privileged fixed 0.5 gold mixture. A7 remains an optional weak-execution control (0.5 mixture). They share the paper interaction and final-token mask. A0 is validation-only. PPO entrypoints use the existing step-level GAE/critic implementation with terminal rewards.

Eq. (9) is computed by `python -m agent_r1.evaluation.summarize`. Its protocol includes parsed local checks and action validity, including missing required evidence. Code behavior logs include revision count, repeated tests of the same artifact, and revisions after an earlier failed developer test. These are descriptive statistics, not a classifier of semantic self-correction.

Consistency audits use `paper-local-consistency-v2`. Version 1 also gated C_d on final-submission eligibility; those records must be recomputed under the current definition before using the summary tool. The summary rejects older audit versions rather than mixing metric definitions.

TACO preparation preserves source shard/row order after task eligibility filtering and deterministic validation holdout. Existing cached prepared datasets must be regenerated to use this ordering. Paper benchmark releases/checkpoints and historical numerical results cannot be reconstructed from implementation changes alone.
