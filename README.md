# SparkLightGBM

SparkLightGBM is a small, Python-only bridge between Apache Spark DataFrames and the official [LightGBM](https://github.com/lightgbm-org/LightGBM) Python package. It provides Spark ML-style estimators for classification, regression, quantile regression, and LambdaRank without requiring SynapseML or a Scala/JVM extension.

## Status

Version `0.1.0b1` is a Beta release for objective runtime testing. It is not production-stable, and the future stable API/release is `1.0.0`. The public estimator and model APIs are intentionally small so the execution layer can evolve without coupling callers to implementation details.

```python
from sparklightgbm import LightGBMClassifier
model = LightGBMClassifier(features_col="features", label_col="label", n_estimators=100)
predictions = model.fit(train_df).transform(test_df)
```

Install with `pip install sparklightgbm pyspark`. SparkLightGBM never calls `toPandas()`; rows are read from Spark partitions and the learner remains official LightGBM. With the default `num_workers=1`, partition data is collected to the driver, so this mode is intended for local and smaller workloads. Set `num_workers` above one to coordinate LightGBM's native data-parallel learner through Spark barrier execution; configure `local_listen_port` when the cluster requires a different executor port range.

Distributed mode requires LightGBM and this package on every executor, reachable executor hostnames/IPs and the listener ports, compatible native LightGBM builds, and a cluster that supports Spark barrier execution. Worker hostname resolution is environment-dependent across local Spark, Databricks, and standard clusters; configure executor DNS or `SPARK_LOCAL_IP` as needed. `validation_data` and early stopping currently require `num_workers=1`. The bridge supports Spark 3.3+, Python 3.9+, native model persistence, and the estimator capabilities listed above. See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).
