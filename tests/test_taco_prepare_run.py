import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TacoPrepareRunTest(unittest.TestCase):
    """Both TACO arms accept the 4B (20) and 9B (8) paper batch sizes on eight GPUs."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in ("train.parquet", "validation.parquet"):
            pq.write_table(pa.table({"question": ["q"] * 4}), self.root / name)
        (self.root / "sidecar.jsonl").write_text("{}\n")
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, module, batch_size, *extra):
        output = self.root / f"{module.rsplit('.', 2)[-2]}-{batch_size}"
        command = [sys.executable, "-m", module, "--project-dir", str(PROJECT_ROOT), "--output-dir", str(output),
                   "--model-path", str(self.model), "--train-path", str(self.root / "train.parquet"),
                   "--validation-path", str(self.root / "validation.parquet"),
                   "--sidecar-path", str(self.root / "sidecar.jsonl"), "--num-gpus", "8",
                   "--train-max-samples", "4", "--train-batch-size", str(batch_size), "--rollout-n", "4",
                   "--total-training-steps", "1", "--total-epochs", "1", "--save-freq", "1", "--seed", "42", *extra]
        return subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)

    def test_both_arms_accept_4b_and_9b_batches_and_reject_others(self):
        for module, extra in (("recipes.taco_a9.prepare_run", ("--terminal-warmup-steps", "0")),
                              ("recipes.taco_a1.prepare_run", ())):
            for batch_size, accepted in ((20, True), (8, True), (16, False)):
                with self.subTest(module=module, batch_size=batch_size):
                    result = self.prepare(module, batch_size, *extra)
                    self.assertEqual(result.returncode == 0, accepted, result.stderr[-800:])


if __name__ == "__main__":
    unittest.main()
