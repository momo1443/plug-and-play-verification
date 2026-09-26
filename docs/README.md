# Repository guide

The repository is organized around the paper's four domains. Run commands from
the repository root; [the launcher index](../scripts/README.md) lists the entrypoints.

```text
agent_r1/          Shared verifier contract, rewards, rollout, GRPO/PPO, metrics
  config/          Shared Hydra trainer configuration
recipes/           Domain flows, verifiers, data preparation, domain YAML configs
scripts/           Training and evaluation entrypoints, grouped by domain
  common/          Shared model adaptation and paper budgets
  extensions/      Optional ToolEnv/A8 experiments and hardware/debug profiles
  patches/         Explicit compatibility patches for the training environment
tests/             CPU contracts, launcher checks, and runtime-dependent tests
assets/images/     The two plug-and-play verifier figures
docs/              Implementation notes, writing drafts, and historical records
verl_patches/      Runtime weight-transfer implementation
```

Domain YAML files stay beside the code that consumes them; shared trainer config
stays in `agent_r1/config/`. The `agent_r1` and `recipes` Python package names are
preserved so imports, Hydra targets, and checkpoint configurations remain stable.

## Current documentation

- [Paper implementation contract](paper-contract.md): reward semantics, training profiles, and validation limits.
- [Frozen process judge](experiments/llm-process-judge.md): comparison-arm protocol.
- [Shared launcher settings](../scripts/common/README.md): model adaptation and overrides.
- [DeepScaleR](../recipes/deepscaler/README.md) and [HotpotQA](../recipes/hotpotqa/README.md): task-specific setup.
- [README writing draft](drafts/README.md): preserved draft, separate from the root README.
- [Historical records](archive/README.md): plans, progress, and old setup notes.

`recipes/hotpotqa_lr/` also supplies parsing, answer normalization, and artifact
helpers used by the current HotpotQA/TACO flows and validation recorder. It is
retained even though A8 launchers are optional extensions. `recipes/hotpotqa/`
provides shared retrieval, evidence, and judge services. `recipes/mathvision/`
and `recipes/livecodebench_a0/` contain evaluation utilities.

## Paths after cleanup

| Previous location | Current location |
| --- | --- |
| `examples/<domain>/` | `scripts/<domain>/` (math uses `scripts/deepscaler/`) |
| `examples/hotpotqa_a9/run_validation.sh` | `scripts/hotpotqa/run_validation.sh` |
| ToolEnv, A8, hardware/debug wrappers | `scripts/extensions/` |
| `scripts/patch_*.py` | `scripts/patches/` |
| `scripts/finegrained_eval.py` | `scripts/extensions/hotpotqa_lr/finegrained_eval.py` |
| `image/` paper figures | `assets/images/` |
| `README_DRAFT.txt` | `docs/drafts/README.md` |
| `docs/plan/`, `docs/progress/`, old `CLAUDE.md` | `docs/archive/` |

The unused GSM8K examples (whose recipe was absent), independent Paper Search
recipe/examples, upstream Agent-R1 documentation site/results/figures, and two
obsolete DeepScaleR aliases were removed. Use `run_deepscaler_a9_uniform.sh`
instead of `run_deepscaler_a9_fixed73.sh`, and the extension's
`run_deepscaler_tool_a1.sh` instead of `run_deepscaler_tool.sh`.

Their last pre-cleanup versions remain in Git commit
`2fb3962a61c6aafb962dbe66c958b00e7149bd87`. For example:

```bash
git show 2fb3962:recipes/paper_search/README.md
```

Model checkpoints, datasets, and experiment outputs are external or ignored.
The cleanup does not migrate or remove them. Existing external job scripts must
update their `examples/` paths using the table above.
