# PNP tests

This directory maintains the current four PNP tasks (DeepScaleR math, HotpotQA,
TACO, and visual reasoning), AIME agent evaluation, and the shared training and
rollout infrastructure. Historical A0/A8 experiments, LLM judge comparisons,
ToolEnv extensions, and tests for their launchers are excluded.

Run from the repository root in the `pnp` environment:

```bash
CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:tests \
  python -m pytest -q tests
```

- `test_pnp_launcher.py` and `test_pnp_config.py` exercise real shell argument
  forwarding with a Python stub; they do not start training.
- `test_pnp_flows.py`, `test_math_continuation.py`, and `test_pnp_contracts.py`
  execute production flows using mocked generation/execution boundaries.
- `test_pnp_evaluation.py` checks shared budgets, complete denominators, hidden
  reference isolation, separate final grading, and evaluation wrappers.
- `test_official_graders.py` checks pinned upstream checksums, official answer
  extraction and equality, Decimal stdio comparison, functional results, and
  sandbox forwarding. Install `scripts/evaluation/setup_official_graders.sh`
  before running these tests.
- `pnp_test_utils.py` loads production definitions while replacing heavy runtime
  imports. It is a shared fixture module, not a test suite.
- Domain verifier tests preserve regressions for missing submissions, failed
  checks, excluded inapplicable checks, replay, and independent outcome rewards.
- Other retained suites cover data preparation, answer extraction, rollout logs,
  worker behavior, synchronization, GRPO advantages, and validation metrics.

These CPU checks do not establish GPU training stability or model accuracy.
