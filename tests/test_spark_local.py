import os
import sys

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
lightgbm = pytest.importorskip("lightgbm")


def test_classifier_spark_local_matches_native_prediction():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession
    from sparklightgbm import LightGBMClassifier, LightGBMRanker, LightGBMRegressor

    spark = SparkSession.builder.master("local[2]").appName("sparklightgbm-test").config("spark.ui.enabled", "false").getOrCreate()
    try:
        frame = spark.createDataFrame([([0.0, 0.0], 0.0), ([0.0, 1.0], 0.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 1.0)], ["features", "label"])
        model = LightGBMClassifier(n_estimators=5, num_leaves=4, seed=7).fit(frame)
        result = model.transform(frame).select("prediction", "probability", "rawPrediction", "leafPrediction").collect()
        assert len(result) == 4
        features = [[row[0][0], row[0][1]] for row in frame.select("features").collect()]
        native_scores = model.booster.predict(features, raw_score=True)
        assert all(row.probability is not None for row in result)
        assert all(row.rawPrediction is not None for row in result)
        assert all(row.leafPrediction is not None and all(value is not None for value in row.leafPrediction) for row in result)
        assert all(0.0 <= row.probability[0] <= 1.0 for row in result)
        assert all(float(row.prediction) == float(np.argmax(row.rawPrediction)) for row in result)
        assert all(isinstance(value, int) for row in result for value in row.leafPrediction)
        expected_raw = np.column_stack((-native_scores, native_scores))
        np.testing.assert_allclose([row.rawPrediction for row in result], expected_raw, rtol=1e-12, atol=1e-12)
        expected_probability = 1.0 / (1.0 + np.exp(-expected_raw))
        np.testing.assert_allclose([row.probability for row in result], expected_probability, rtol=1e-12, atol=1e-12)
        assert model.predict_probability([[1.0, 1.0]]).shape == (1, 2)
        assert model.predict_raw([[1.0, 1.0]]).shape == (1, 2)
        assert model.predict_leaf([[1.0, 1.0]]).shape[0] == 1
        assert model.predict_leaf([[1.0, 1.0]]).shape[1] >= 1
        assert model.predict_shap([[1.0, 1.0]]).shape == (1, 3)

        null_features = spark.createDataFrame([(None,)], frame.select("features").schema)
        null_result = model.transform(null_features).select("prediction", "probability", "rawPrediction", "leafPrediction").first()
        assert all(value is None for value in null_result)

        multiclass = spark.createDataFrame([([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([0.0, 2.0], 2.0), ([1.0, 0.0], 0.0), ([1.0, 1.0], 1.0), ([1.0, 2.0], 2.0)], ["features", "label"])
        multiclass_result = LightGBMClassifier(num_class=3, n_estimators=3, min_data_in_leaf=1, seed=7).fit(multiclass).transform(multiclass).select("prediction", "probability", "rawPrediction", "leafPrediction").collect()
        assert len(multiclass_result) == multiclass.count()
        assert all(row.probability is not None and len(row.probability) == 3 for row in multiclass_result)
        assert all(row.rawPrediction is not None and len(row.rawPrediction) == 3 for row in multiclass_result)
        assert all(row.leafPrediction is not None and all(isinstance(value, int) for value in row.leafPrediction) for row in multiclass_result)
        assert all(abs(sum(row.probability) - 1.0) < 1e-9 for row in multiclass_result)
        assert all(float(row.prediction) == float(np.argmax(row.rawPrediction)) for row in multiclass_result)

        regression = spark.createDataFrame([([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 2.0)], ["features", "label"])
        for estimator in (LightGBMRegressor(n_estimators=3, min_data_in_leaf=1, seed=7), LightGBMRegressor(objective="quantile", alpha=0.5, n_estimators=3, min_data_in_leaf=1, seed=7)):
            regression_result = estimator.fit(regression).transform(regression).select("prediction", "rawPrediction", "leafPrediction").collect()
            assert len(regression_result) == regression.count()
            assert all(row.prediction is not None and row.rawPrediction is not None for row in regression_result)
            assert all(row.leafPrediction is not None and all(isinstance(value, int) for value in row.leafPrediction) for row in regression_result)

        ranking = spark.createDataFrame([([0.0], 0.0, 0), ([1.0], 1.0, 0), ([0.0], 0.0, 1), ([1.0], 1.0, 1)], ["features", "label", "group"])
        ranking_result = LightGBMRanker(group_col="group", n_estimators=3, min_data_in_leaf=1, seed=7).fit(ranking).transform(ranking).select("prediction", "rawPrediction", "leafPrediction").collect()
        assert len(ranking_result) == ranking.count()
        assert all(row.prediction is not None and row.rawPrediction is not None for row in ranking_result)
        assert all(row.leafPrediction is not None and all(isinstance(value, int) for value in row.leafPrediction) for row in ranking_result)
    finally:
        spark.stop()
