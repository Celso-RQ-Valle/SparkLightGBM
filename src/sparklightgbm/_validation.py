from __future__ import annotations
import importlib

def require_runtime():
    try:
        pyspark = importlib.import_module("pyspark")
        lightgbm = importlib.import_module("lightgbm")
    except ImportError as exc:
        raise ImportError("SparkLightGBM requires pyspark and lightgbm on the driver and executors") from exc
    spark_version = tuple(int(x) for x in pyspark.__version__.split(".")[:2])
    if not (3, 4) <= spark_version < (4, 0):
        raise RuntimeError("SparkLightGBM supports PySpark 3.4 through 3.5")
    return pyspark, lightgbm

def check_columns(df, features_col, label_col=None):
    missing = [c for c in [features_col, label_col] if c and c not in df.columns]
    if missing:
        raise ValueError(f"Missing Spark DataFrame columns: {', '.join(missing)}")
