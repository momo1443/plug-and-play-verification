import copy
import unittest

import numpy as np

from agent_r1.trainer.validation_metrics import scalar_validation_metrics
from verl.trainer.ppo.metric_utils import process_validation_metrics


class ValidationMetricsTest(unittest.TestCase):
    def test_audits_are_retained_but_not_averaged(self):
        infos = {
            "reward": [0.0, 1.0],
            "acc": [False, True],
            "process": [np.float32(0.5), np.float32(1.0)],
            "audit": [{"equation": "1+1=2"}, {"equation": "2+2=4"}],
            "credits": [[0.5], [1.0]],
            "reason": ["failed", "passed"],
            "optional": [None, 1.0],
            "empty": [],
        }
        original = copy.deepcopy(infos)
        selected = scalar_validation_metrics(infos)
        self.assertEqual(set(selected), {"reward", "acc", "process"})
        metrics = process_validation_metrics(["math", "math"], ["a", "b"], selected)
        self.assertEqual(metrics["math"]["acc"]["mean@1"], 0.5)
        self.assertEqual(metrics["math"]["process"]["mean@1"], 0.75)
        self.assertEqual(infos, original)

    def test_string_predictions_still_support_majority_voting(self):
        selected = scalar_validation_metrics({"reward": [1.0, 1.0], "pred": ["2", "2"]})
        metrics = process_validation_metrics(["math", "math"], ["a", "a"], selected)
        self.assertEqual(metrics["math"]["reward"]["maj@2/mean"], 1.0)
