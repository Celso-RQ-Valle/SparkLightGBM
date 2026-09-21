import math

import numpy as np

from sparklightgbm.models import LightGBMClassificationModel


class _ClassifierEstimator:
    kind = "classifier"


class _SingleScoreBooster:
    def __init__(self, score):
        self.score = score

    def num_model_per_iteration(self):
        return 1

    def predict(self, matrix, pred_leaf=False, raw_score=False, pred_contrib=False):
        if raw_score:
            return np.asarray([self.score])
        return np.asarray([1.0 / (1.0 + math.exp(-self.score))])


def test_binary_prediction_matches_synapseml_contract():
    references = [
        (0.47733373687589564, [-0.47733373687589564, 0.47733373687589564], [0.3828819217908761, 0.6171180782091239]),
        (-0.48201806746682063, [0.48201806746682063, -0.48201806746682063], [0.6182242986433802, 0.38177570135661976]),
    ]
    for score, expected_raw, expected_probability in references:
        model = LightGBMClassificationModel(_SingleScoreBooster(score), _ClassifierEstimator(), 1)
        np.testing.assert_allclose(model._predict([1.0], "raw"), expected_raw, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(model._predict([1.0], "prob"), expected_probability, rtol=0.0, atol=1e-15)
