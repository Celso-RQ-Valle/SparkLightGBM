# SparkLightGBM

SparkLightGBM is a small, Python-only bridge between Apache Spark DataFrames and the official [LightGBM](https://github.com/lightgbm-org/LightGBM) Python package. It provides Spark ML-style estimators for classification, regression, quantile regression, and LambdaRank without requiring SynapseML or a Scala/JVM extension.

## Status

Version 0.1.0 is an alpha release. The public estimator and model APIs are intentionally small so the execution layer can evolve toward 1.0 without coupling callers to implementation details.

```python
from sparklightgbm import LightGBMClassifier
model = LightGBMClassifier(features_col="features", label_col="label", n_estimators=100)
predictions = model.fit(train_df).transform(test_df)
```

Install with `pip install sparklightgbm pyspark`. SparkLightGBM never calls `toPandas()`; rows are read from Spark partitions and the learner remains official LightGBM. Set `num_workers` above one to coordinate LightGBM's native data-parallel learner through Spark barrier execution. CPU is the default. It has no required SynapseML dependency.

The bridge supports Spark 3.3+, Python 3.9+, native model persistence, validation data, and early stopping. See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).
