from __future__ import annotations
import importlib
import re

def require_runtime():
    try:
        pyspark = importlib.import_module("pyspark")
        lightgbm = importlib.import_module("lightgbm")
    except ImportError as exc:
        raise ImportError("SparkLightGBM requires pyspark and lightgbm on the driver and executors") from exc
    # Vendor and development builds can append suffixes to the release number.
    match = re.match(r"^(\d+)\.(\d+)", pyspark.__version__)
    if match is None or tuple(map(int, match.groups())) < (3, 4):
        raise RuntimeError(f"SparkLightGBM requires PySpark >=3.4; found {pyspark.__version__}")
    return pyspark, lightgbm

def check_columns(df, features_col, label_col=None):
    missing = [c for c in [features_col, label_col] if c and c not in df.columns]
    if missing:
        raise ValueError(f"Missing Spark DataFrame columns: {', '.join(missing)}")
