# Codex Working Agreement

## Working scope

- Unless the user explicitly expands the scope, all work for this project must stay inside `/nas/deepresearch/zsb/corhort/project/agenticrl`.
- Do not create or edit files in sibling projects or in the parent project directory.

## Planning and progress records

- At the beginning of every task-oriented conversation, create or update a Markdown plan in `docs/plan/` before implementing code.
- Name plan files `YYYY-MM-DD.md` or `YYYY-MM-DD-<short-task-name>.md`. Record the goal, scope, implementation steps, validation steps, assumptions, and status, with a separate section for each conversation when a daily file is shared.
- During every day on which work is performed, create or update `docs/progress/YYYY-MM-DD.md`.
- The daily progress file must record completed work, validation results, open issues, and the next action. Append new conversation entries instead of deleting earlier entries from the same day.
- Keep plans and progress truthful and synchronized with the actual repository state. Mark unfinished items explicitly.

These requirements apply throughout the `agenticrl` directory unless a more specific `AGENTS.md` overrides them.

## Project research objective

The project's overarching objective is to determine whether agentic post-training can move
verification from a bolted-on test-time procedure into the policy itself. Train a tool-using model
to make every load-bearing action and answer claim ship with an explicit, structured certificate
that a cheap deterministic program can check, so certificate-complete behavior becomes the
policy's default rather than an occasional result of prompting. The learned behavior should be a
general schema-following and evidence-grounding capability, including transfer to held-out or
previously unseen tool schemas supplied in context, rather than memorization of one tool name or
one benchmark format.

For snapshot-able closed environments, place deterministic, self-computable signals in the reward
function and keep gold-derived or judgment-based signals for evaluation. The reward loop should
not require a trainable reward model or an LLM judge. The long-term anchor is **certificate
sufficiency**: a deterministic re-executor must be able to reproduce the final answer from the
emitted evidence artifacts and declared computation alone. Natural-language chain-of-thought is
not the verification target; externalize the load-bearing step as an artifact and verify that
artifact.

The scientific outcome is not certificate pass rate alone. Jointly measure:

- certificate/schema validity, exact provenance, and deterministic replay;
- task-relevant evidence acquisition and final task quality;
- whether reward-aligned behavior transfers to held-out questions, domains, and unseen tool
  schemas;
- the **legibility tax**, meaning any accuracy, recall, latency, or token-cost loss caused by making
  behavior checkable; and
- **verifier demotion**, meaning how much post-training reduces the need for test-time verification,
  revision, or rejection at matched quality.
