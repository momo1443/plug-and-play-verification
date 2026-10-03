# Paper implementation contract

Target: `2027_PnP_Verifier__arXiv_ (1).pdf`, Sections 3–4 and Appendices B–C. The user-requested multi-turn math protocol below extends the manuscript's single-turn setting; it does not establish new benchmark scores.

- Verifiers return nonnegative, bounded, source-step credits. Reward composition and policy optimization do not interpret domain artifacts.
- Outcome reward is placed at the last policy step. Process credits remain at their source steps. For the proposed verifier and judge arms, the first 50 updates use outcome only; subsequent updates share one reproducible uniform weight per prompt group. Validation uses outcome only.
- Terminal eligibility requires a final submission in the domain protocol, regardless of whether that answer is correct or its certificate passes. An incomplete trajectory receives zero composed training reward.
- GRPO uses undiscounted returns, prompt/step grouping, sample standard deviation, and the Appendix B singleton convention. Policy and KL losses use policy tokens only.
- Within each domain, outcome-only, verifier, and frozen-judge arms share prompts, tools, final-answer extraction, generation limits, and model adaptation.
- Formal runs select the first 10,000 rows of the prepared training split and use 500 updates, 4 rollouts/prompt, seed 42, learning rate 1e-6, and actor-loss KL coefficient 0.001. Per the updated hardware configuration, 4B uses prompt batch 20 and 9B uses batch 8. At 500 updates these sample 10,000 and 4,000 prompts respectively, with four rollouts each; the selected training prefix remains 10,000 rows. Manifests record selection size and actual sampled prompt count separately. Appendix C must reflect the 9B batch revision.

| Domain | 4B batch | 9B batch | Prompt tokens | Completion tokens | Policy steps |
|---|---:|---:|---:|---:|---:|
| DeepScaleR | 20 | 8 | 2048 initial; 8192 history | 4096 (4B), 5120 (9B), total per trajectory | 5 |
| HotpotQA | 20 | 8 | 8192 | 1024 | 4 |
| TACO | 20 | 8 | 8192 | 2048 | 5 |
| Vision-R1 | 20 | 8 | 8192 | 2048 | 3 |

9B uses LoRA rank/alpha 64 and the seven projection modules listed in Appendix C; 4B uses full fine-tuning. DeepScaleR now permits up to five reasoning generations by default (`DEEPSCALER_MAX_STEPS`), with a shared total token budget and per-turn cap `ceil(total / max_steps)`. A reference-independent extracted final answer ends the trajectory; incomplete trajectories retain their steps/masks but receive zero composed reward. Equation/judge verification occurs retrospectively and credits remain on their source turns. No correctness or verification feedback enters model context. Set `DEEPSCALER_MAX_STEPS=1` for the original single-turn setting. The mathematics paragraph in Appendix B and the corresponding Appendix C protocol must reflect this extension before associating results with it. The answer-checking ToolEnv remains a separate optional protocol.

Evaluation extracts one final answer independently of the reference, then scores it. Equation (9) uses all applicable checks, including failures and missing required evidence, under a fixed protocol across policies. Audit metrics do not affect optimizer rewards. Revision and retesting are recorded separately from strict consistency.

Implementation map: `agent_r1/verifier` owns contracts and composition; domain recipes own checks and adapters; `agent_r1/trainer/ppo` owns advantages/objectives; `scripts/*` owns runnable profiles; `agent_r1/evaluation` and `scripts` own offline metrics. CPU checks cover answer extraction, contracts, real launcher arguments, and agent-flow wiring. GPU training and benchmark scores require separate runs and are not implied by passing tests.

HotpotQA gold ablations retain their explicit reward choices: A2 is process-only, A3 is a fixed 0.5 mixture. A7 remains an optional weak-execution control (0.5 mixture), not a gold-evidence arm. They share the paper interaction and final-token mask. A0 is validation-only. PPO entrypoints use the existing step-level GAE/critic implementation with terminal rewards.

Eq. (9) is computed by `python -m agent_r1.evaluation.summarize`. Its protocol includes parsed local checks and action validity, including missing required evidence. Code behavior logs include revision count, repeated tests of the same artifact, and revisions after an earlier failed developer test. These are descriptive statistics, not a classifier of semantic self-correction.

TACO preparation preserves source shard/row order after task eligibility filtering and deterministic validation holdout. Existing cached prepared datasets must be regenerated to use this ordering. Paper benchmark releases/checkpoints and historical numerical results cannot be reconstructed from implementation changes alone.
