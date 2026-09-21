import os
import sys

import pytest

pyspark = pytest.importorskip("pyspark")
lightgbm = pytest.importorskip("lightgbm")


def test_classifier_spark_local_matches_native_prediction():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import SparkSession
    from sparklightgbm import LightGBMClassifier

    spark = SparkSession.builder.master("local[2]").appName("sparklightgbm-test").config("spark.ui.enabled", "false").getOrCreate()
    try:
        frame = spark.createDataFrame([(Vectors.dense([0.0, 0.0]), 0.0), (Vectors.dense([0.0, 1.0]), 0.0), (Vectors.dense([1.0, 0.0]), 1.0), (Vectors.dense([1.0, 1.0]), 1.0)], ["features", "label"])
        model = LightGBMClassifier(n_estimators=5, num_leaves=4, seed=7).fit(frame)
        result = model.transform(frame).select("prediction", "probability", "rawPrediction", "leafPrediction").collect()
        assert len(result) == 4
        assert model.predict_probability([[1.0, 1.0]]).shape == (1,)
        assert model.predict_raw([[1.0, 1.0]]).shape == (1,)
        assert model.predict_leaf([[1.0, 1.0]]).shape[0] == 1
        assert model.predict_leaf([[1.0, 1.0]]).shape[1] >= 1
        assert model.predict_shap([[1.0, 1.0]]).shape == (1, 3)
    finally:
        spark.stop()
