"""Reproducible SparkLightGBM scaling benchmark; emits machine-readable JSON."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def timed(action):
    started = time.perf_counter()
    value = action()
    return value, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--features", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--partitions", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import lightgbm
    import pyspark
    from pyspark.sql import SparkSession, functions as F
    from sparklightgbm import LightGBMClassifier

    spark = SparkSession.builder.appName("SparkLightGBMBenchmark").getOrCreate()
    partitions = args.partitions or max(args.workers, spark.sparkContext.defaultParallelism)

    def generate():
        base = spark.range(args.rows, numPartitions=partitions)
        features = F.transform(F.sequence(F.lit(0), F.lit(args.features - 1)), lambda i: ((F.col("id") * (i + 3)) % 997).cast("double") / 997.0)
        return base.select(features.alias("features"), ((F.col("id") * 17) % 2).cast("double").alias("label")).cache()

    frame, generation_seconds = timed(generate)
    row_count, materialization_seconds = timed(frame.count)
    estimator = LightGBMClassifier(num_workers=args.workers, n_estimators=args.iterations, seed=7)
    model, training_seconds = timed(lambda: estimator.fit(frame))
    predictions = model.transform(frame).select("prediction")
    prediction_sum, prediction_seconds = timed(lambda: predictions.agg(F.sum("prediction")).first()[0])
    model_size = len(model.booster.model_to_string().encode("utf-8"))

    result = {
        "configuration": vars(args) | {"output": str(args.output), "partitions": partitions},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "spark": pyspark.__version__,
            "lightgbm": lightgbm.__version__,
            "master": spark.sparkContext.master,
            "default_parallelism": spark.sparkContext.defaultParallelism,
        },
        "measurements": {
            "rows": row_count,
            "generation_seconds": generation_seconds,
            "materialization_seconds": materialization_seconds,
            "training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds,
            "prediction_rows_per_second": row_count / prediction_seconds,
            "prediction_checksum": prediction_sum,
            "model_bytes": model_size,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    spark.stop()


if __name__ == "__main__":
    main()
