# Qwen3.5-9B process judge

Status: implemented and CPU-validated; real GPU serving/training remains unverified.

Goal: make the four-domain judge baseline retain deterministic binary outcome rewards while a frozen Qwen3.5-9B supplies step-attributed process rewards.

Scope: this user-selected local experiment/agentic-rlvr checkout. The historical NAS path in CLAUDE.md is superseded by the user's explicit local scope.

Implementation:
- Keep HotpotQA A6 evidence-coverage rewards and 0.5/0.5 mixing; align standalone judge defaults with Qwen3.5-9B.
- Replace terminal judge scoring in math/code/vision with causal intermediate-step scoring returning VerificationResult.
- Normalize each intermediate score by the configured maximum agent turns; omit final submissions, hidden answers and private test results from judge inputs.
- Reuse each domain's existing outcome-only warmup and group-shared mixing schedule. Retain process credit on failed/incomplete trajectories, with binary outcome zero when no final answer exists.
- Send actual root/crop images to the vision judge; include images in the request cache identity.
- Preserve fail-closed behavior on judge transport/parse failures; math/code/vision skip judge calls during validation/warmup. HotpotQA retains validation-only judge logging, excluded from rewards.
- Update manifests, documentation, and regression tests.

Validation: focused helper/backend tests and mocked flow integration tests for reward placement, outcome preservation, missing submissions, warmup, validation, and failures. Compile modified Python and syntax-check shell launchers. No GPU training or result-table claims without new runs.

Completed: all implementation steps above; 70 focused tests passed, 15 modified Python files compiled, changed YAML parsed with explicit warmup settings, shell launchers syntax-checked, and all four configuration-only launcher paths passed. Critical F/E9 lint passed on changed implementation/test files, apart from three pre-existing unused imports in the HotpotQA flow (only a model-name comment changed there). The broader existing verifier test could not import the absent Hydra/Ray/veRL training stack. See the daily progress record for exact validation scope.
