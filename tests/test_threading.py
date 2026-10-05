from types import SimpleNamespace

import numpy as np
import pytest

from sparklightgbm import LightGBMRegressor
from sparklightgbm.models import BaseLightGBMModel, _score_partition


@pytest.mark.parametrize("thread_params", [{}, {"num_threads": 0}, {"num_threads": 1}, {"num_threads": 3}, {"n_jobs": 2}])
@pytest.mark.parametrize("kind,binary", [("regressor", False), ("classifier", True), ("classifier", False), ("ranker", False)])
def test_all_partition_predictions_receive_thread_policy(monkeypatch, thread_params, kind, binary):
    calls = []

    def predict(matrix, **kwargs):
        calls.append(kwargs)
        if kwargs.get("pred_leaf"):
            return np.zeros((len(matrix), 2))
        if kind == "classifier" and not binary:
            return np.zeros((len(matrix), 3))
        return np.zeros(len(matrix))

    monkeypatch.setattr("sparklightgbm.models._cached_booster", lambda _: SimpleNamespace(predict=predict))
    model = BaseLightGBMModel(None, LightGBMRegressor(**thread_params), 1)
    # Each fitted model retains its policy even if the estimator is reused.
    model.estimator.params["num_threads"] = 99
    result = list(_score_partition(
        iter([([1.0],), (None,), ([2.0],)]), "model", 0, [0], 1,
        kind, binary, True, kind == "classifier", True, model._thread_params,
    ))
    assert len(result) == 3
    assert result[1][1:] == (None,) * (4 if kind == "classifier" else 3)
    expected = thread_params or {"num_threads": 0}
    assert calls
    for call in calls:
        assert {k: v for k, v in call.items() if k not in {"raw_score", "pred_leaf"}} == expected


def test_loaded_model_accepts_prediction_thread_override(monkeypatch):
    monkeypatch.setattr("sparklightgbm.models.load_native_model", lambda _: SimpleNamespace(num_feature=lambda: 1))
    assert BaseLightGBMModel.load_native_model("model.txt")._thread_params == {}
    assert BaseLightGBMModel.load_native_model("model.txt", num_threads=2)._thread_params == {"num_threads": 2}
