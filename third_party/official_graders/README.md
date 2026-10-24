# Official benchmark graders

These upstream files are copied without changes. `manifest.json` records their
source repositories, fixed commits, and SHA-256 checksums. Each copy retains its
upstream license. Update by replacing the complete upstream file and recording
the new revision and checksum; do not edit grading rules locally.

- HotpotQA: `hotpot_evaluate_v1.py`, answer EM and token F1.
- MATH-Vision: `evaluation/evaluate.py` and `evaluation/utils.py`, official
  answer extraction and mathematical equivalence.
- LiveCodeBench: `lcb_runner/evaluation/testing_util.py`, stdio and functional
  grading. Submitted code executes inside the existing bubblewrap boundary.

Run `bash scripts/evaluation/setup_official_graders.sh` from the repository root.
The isolated environment avoids changing the training environment's SymPy or
ANTLR dependencies. Set `PNP_GRADER_PYTHON` to use another environment containing
the same dependencies.

AIME2026 uses exact equality against the dataset's official integer answers.
PNP selects the final submission independently of the reference before comparing
its integer value; this selection is part of the agent interaction protocol.
