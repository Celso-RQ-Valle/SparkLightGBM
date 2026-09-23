# SparkLightGBM

SparkLightGBM is a lightweight bridge between Apache Spark DataFrames and the official [LightGBM](https://github.com/lightgbm-org/LightGBM) Python package. It supports native LightGBM training across Spark executors without requiring SynapseML or a Scala/JVM extension.

The project aims for efficient training and inference, numerical correctness, distributed scalability, and production reliability while keeping installation simple, dependencies minimal, and behavior portable across Spark environments. It does not aim to reproduce every feature of larger Spark integrations. Simplicity, performance, compatibility, and predictable behavior take priority over feature count.

## Status

`0.9.x` is the pre-1.0 real-world validation series. It is intended for practical evaluation, but the public API may still change based on results across different environments, Spark configurations, datasets, and workloads. Version `1.0.0` is reserved for the stable release after that validation.

## Installation

```bash
pip install "sparklightgbm[spark]"
```

Use `pip install sparklightgbm` when PySpark is already supplied by a managed Spark environment. The supported compatibility policy for `0.9.x` is Python 3.9-3.12, PySpark 3.4-3.5, LightGBM 4.x or newer, and NumPy 1.21 or newer. PySpark, LightGBM, NumPy, and this package must be available on the driver and every executor. Linux and Windows are exercised in CI; distributed multi-node training is primarily expected on Linux clusters.

The optional development installation is:

```bash
pip install "sparklightgbm[dev]"
```

SparkLightGBM never calls `toPandas()` and does not reimplement LightGBM. CPU is the default. There is no required SynapseML dependency.

## Quick start

The `features_col` value must identify a Spark column containing a numeric feature vector or a numeric array/list. Spark ML `Vector` values and ordinary Spark arrays are accepted. The label column must contain numeric labels or targets.

```python
from sparklightgbm import LightGBMClassifier

estimator = LightGBMClassifier(
    features_col="features",
    label_col="label",
    num_workers=None,
    num_iterations=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_data_in_leaf=20,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=1.0,
    min_gain_to_split=0.0,
    max_bin=255,
    seed=7,
    feature_fraction_seed=7,
    bagging_seed=7,
    data_random_seed=7,
)

model = estimator.fit(train_df)
predictions = model.transform(test_df)
predictions.select("prediction", "probability").show()
```

## Estimators and input parameters

All three estimators accept the shared parameters below. Additional keyword arguments are forwarded to native LightGBM, allowing supported LightGBM parameters to be used without waiting for a wrapper release.

| Parameter | Accepted type | Default | Description |
| --- | --- | --- | --- |
| `features_col` | `str` | `"features"` | Spark column containing a numeric vector or array. |
| `label_col` | `str` | `"label"` | Spark column containing the target. |
| `prediction_col` | `str` | `"prediction"` | Output column for the predicted value or class. |
| `raw_prediction_col` | `str \| None` | `"rawPrediction"` | Output column for raw LightGBM scores; `None` omits it. |
| `probability_col` | `str \| None` | `"probability"` | Classifier probability output column; `None` omits it. |
| `leaf_prediction_col` | `str \| None` | `"leafPrediction"` | Output column containing integer leaf indices; `None` omits it. |
| `weight_col` | `str \| None` | `None` | Optional Spark column containing per-row weights. |
| `group_col` | `str \| None` | `None` | Optional Spark column containing ranking group identifiers; required by `LightGBMRanker`. |
| `categorical_feature` | `list[int] \| str \| None` | `None` | LightGBM categorical feature indices or native categorical setting. Feature values must be numeric category codes. |
| `validation_data` | Spark `DataFrame` \| `None` | `None` | Validation DataFrame with matching feature, label, weight, and group names. Can also be passed to `fit()`. |
| `early_stopping_rounds` | `int \| None` | `None` | Rounds without validation improvement; requires validation data. Distributed mode supports exactly aggregatable metrics. |
| `seed` | `int` | `0` | Seed passed to LightGBM. |
| `num_workers` | `int \| None` | `None` | Automatically selects a conservative worker count. Local Spark and ranking use driver training. Clustered classification and regression use the smallest of four workers, the input partition count, and available Spark slots minus one. Set a positive integer to override the strategy. |
| `local_listen_port` | `int` | `12400` | Distributed listener base port; worker `n` uses `local_listen_port + n`. |
| `prediction_batch_size` | `int` | `1024` | Rows scored per native LightGBM prediction batch in each Spark partition. |
| `objective` | `str \| None` | `None` | Native objective: defaults to `binary`/`multiclass`, `regression`, or `lambdarank`; `quantile` is supported for regression. |
| `**params` | Native value | - | Parameters forwarded to `lightgbm.Dataset` or `lightgbm.train`, as applicable. |

Estimator-specific parameters:

| Estimator | Parameter | Accepted type / default | Description |
| --- | --- | --- | --- |
| `LightGBMClassifier` | `num_class` | `int \| None`, default `None` | Number of classes; if omitted, inferred from labels. |
| `LightGBMRegressor` | - | - | Supports native regression parameters and `objective="quantile"`; set native `alpha` for the target quantile. |
| `LightGBMRanker` | - | - | Uses LambdaRank by default. `group_col` identifies groups; rows are sorted by group before native training. |

## Common native LightGBM parameters

The following are commonly used native parameters, not a separate SparkLightGBM parameter system. Their appropriate values depend on the data, objective, validation strategy, and resource constraints.

| Parameter | What it controls |
| --- | --- |
| `num_iterations` | Maximum number of boosting rounds. `n_estimators` and `num_boost_round` are also accepted by SparkLightGBM. |
| `learning_rate` | Contribution of each new tree; interacts with the number of iterations. |
| `num_leaves` | Maximum leaves per tree and therefore much of the model's capacity. |
| `max_depth` | Optional tree-depth limit; negative values leave depth unconstrained. |
| `min_data_in_leaf` | Minimum observations allowed in a leaf, controlling leaf granularity and regularization. |
| `feature_fraction` | Fraction of features considered for each tree. |
| `bagging_fraction` | Fraction of rows used when bagging is active. |
| `bagging_freq` | Frequency of bagging; `0` disables it. |
| `lambda_l1` | L1 regularization applied to leaf weights. |
| `lambda_l2` | L2 regularization applied to leaf weights. |
| `min_gain_to_split` | Minimum gain required to create a split. |
| `max_bin` | Maximum histogram bins used for numeric features; affects accuracy, memory, and speed. |
| `is_unbalance` | Enables automatic binary-class imbalance handling. Do not combine it with `scale_pos_weight`. |
| `scale_pos_weight` | Explicit positive-class weight for binary classification. Do not combine it with `is_unbalance`. |
| `seed` | Top-level seed used by SparkLightGBM and passed to LightGBM. |
| `data_random_seed` | Seed used while constructing histogram bins. |
| `feature_fraction_seed` | Seed used for feature subsampling. |
| `bagging_seed` | Seed used for row subsampling. |
| `drop_seed` | Seed used by DART boosting. |
| `deterministic` | Requests stable CPU results; LightGBM may require related parameters for fully reproducible runs. |

`fit(params={...})` can supply or override native parameters for a training call:

```python
from sparklightgbm import LightGBMRegressor

model = LightGBMRegressor(features_col="features", label_col="target").fit(
    train_df,
    params={"objective": "quantile", "alpha": 0.9, "n_estimators": 200, "learning_rate": 0.03},
)
```

SparkLightGBM documents only common parameters. Availability, aliases, interactions, and exact semantics are determined by the installed LightGBM version; see the [official LightGBM parameter documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html) for the complete reference. SparkLightGBM uses `lightgbm.Dataset` and `lightgbm.train`, not the scikit-learn wrapper.

The public training signature is `fit(dataset, params=None, validation_data=None)`: `dataset` and `validation_data` are Spark `DataFrame` objects, and `params` is a `dict[str, object]` of native LightGBM parameter overrides. A `validation_data` argument passed to `fit()` takes precedence over the constructor value.

## Spark input and output

Training reads rows from Spark partitions. Supported training columns are:

- `features_col`: numeric Spark ML vector, numeric array, or numeric Python list.
- `label_col`: numeric binary/multiclass label, regression target, or ranking relevance score.
- `weight_col`: optional numeric row weight.
- `group_col`: optional numeric/integer ranking group identifier.

Missing feature values should use LightGBM-compatible values such as `NaN`. A null feature vector is skipped; a null label or malformed feature value raises an error during collection or native training.

`model.transform(df)` preserves the input DataFrame and adds configured output columns. It creates `prediction` for every estimator, raw scores when enabled, leaf indices when enabled, and probabilities for classifiers when enabled. Output names can be overridden per call:

The transform signature is `transform(dataset, prediction_col=None, raw_prediction_col=None, probability_col=None, leaf_prediction_col=None)`. Every column-name override is a `str` or `None`; `probability_col` is used only for classifier models.

```python
predictions = model.transform(
    test_df,
    prediction_col="score",
    raw_prediction_col="raw_score",
    probability_col="class_probability",
    leaf_prediction_col="leaf_index",
)
```

## Validation and early stopping

Use a Spark DataFrame with matching column names:

```python
model = LightGBMRegressor(n_estimators=500, early_stopping_rounds=30).fit(
    train_df,
    validation_data=validation_df,
)
```

Validation data remains partitioned in distributed mode. Each worker builds local training and validation datasets, while Spark barrier synchronization aggregates decomposable native metrics before the early-stopping callback. Metrics whose exact global value cannot be reconstructed from shard-level scalar results (currently `auc`, `average_precision`, `map`, and `ndcg`) are rejected for distributed early stopping; use `num_workers=1` for those metrics.

## Local and distributed execution

The worker setting changes where training data is materialized:

| Setting | Execution and memory behavior | Intended use |
| --- | --- | --- |
| `num_workers=1` | Spark partitions are read and the complete training dataset—and validation dataset, when present—is collected into NumPy arrays on the driver. Native LightGBM trains in the driver process. | Local development, compatibility fallback, ranking, and datasets that safely fit in driver memory. |
| `num_workers>1` | Spark repartitions the data into a barrier stage. Each executor-side worker converts only its shard to contiguous NumPy buffers and participates in LightGBM's native `data_parallel` network. Training and validation rows are not collected to the driver; the driver receives bounded metadata and the trained model artifact. | Cluster datasets that should remain distributed. Each worker shard must fit in that executor's memory. |
| `num_workers=None` | Selects `1` for local Spark and ranking. On a cluster, classification and regression reserve one slot and select the smallest of `4`, the remaining Spark parallelism, and the input partition count. The result is always at least `1`. | Portable, conservative default that avoids claiming every reported slot. Set an explicit value when cluster capacity or scheduling calls for another count. |

`SparkContext.defaultParallelism` is an estimate, not a dynamic cluster-capacity reservation. On shared or autoscaling clusters, set `num_workers` explicitly when the scheduler policy requires a specific limit. An explicit value always takes precedence, including in local mode; the requested barrier tasks must be schedulable concurrently.

Distributed workers default to one native LightGBM thread per Spark task to avoid CPU oversubscription. Pass `num_threads` explicitly when the Spark resource configuration provides additional CPU capacity per task.

Distributed execution requires:

- SparkLightGBM, NumPy, and the same compatible LightGBM build on every executor.
- Spark barrier execution support.
- Executor hostnames or IP addresses resolvable and reachable from every other executor.
- Listener ports available between executors; set `local_listen_port` if `12400 + worker_id` is unavailable.
- Correct executor networking. `SPARK_LOCAL_IP` can provide the advertised worker address when hostname resolution is unsuitable.

Inference uses partition-level NumPy batches instead of row-wise Python UDFs. A native booster is cached per reused Python worker, and classification prediction, probability, and raw prediction are derived from the same raw-score batch.

## Current limitations

- The `0.9.x` API is in pre-1.0 validation and may change before `1.0.0`.
- Distributed ranking is not yet supported because query groups must remain complete and worker-local. Ranking therefore uses the single-worker path.
- Distributed early stopping supports metrics that can be exactly aggregated from worker-level results. Non-decomposable metrics currently rejected for this mode include `auc`, `average_precision`, `map`, and `ndcg`; use `num_workers=1` when early stopping depends on them.
- Distributed jobs require Spark barrier scheduling plus stable, mutually reachable executor addresses and ports. Executor loss aborts the coordinated native training job.
- Models use native LightGBM persistence rather than Spark ML `MLWriter`/`MLReader`, and the estimators are not yet Spark ML `Estimator`/`Model` stages for `Pipeline` or `CrossValidator`.
- CPU is the supported default execution path; GPU execution is not currently documented or tested by this project.

## Predictions and explainability

Fitted models expose native LightGBM outputs:

```python
raw = model.predict_raw([[1.0, 2.0]])
probability = model.predict_probability([[1.0, 2.0]])  # classifiers
leaves = model.predict_leaf([[1.0, 2.0]])
shap = model.predict_shap([[1.0, 2.0]])
split_importance = model.feature_importance("split")
gain_importance = model.feature_importance("gain")
```

`predict_shap()` requests native LightGBM contribution values. The output includes one value per feature plus the expected-value contribution. `feature_importance()` accepts `"split"` or `"gain"`.

## Native model persistence

Models are saved in LightGBM's native format. A JSON sidecar containing bridge metadata is written next to the model file:

```python
model.save_native_model("artifacts/model.txt")

from sparklightgbm import LightGBMClassificationModel
restored = LightGBMClassificationModel.load_native_model(
    "artifacts/model.txt",
    features_col="features",
)
```

The native model file is portable across Spark jobs with a compatible LightGBM installation. It is not a Spark ML `PipelineModel` format.

`save_native_model(path)` accepts a filesystem path (`str` or `pathlib.Path`). `LightGBMClassificationModel.load_native_model(path, **kwargs)`, `LightGBMRegressionModel.load_native_model(path, **kwargs)`, and `LightGBMRankingModel.load_native_model(path, **kwargs)` accept the native model path plus optional `features_col`, `prediction_col`, `raw_prediction_col`, `probability_col`, `leaf_prediction_col`, and `kind` strings used to reconstruct Spark output behavior.

## Compatibility and errors

SparkLightGBM validates required columns and checks that supported PySpark and LightGBM installations are available on the driver. The `0.9.x` compatibility policy is Python 3.9-3.12 and PySpark 3.4-3.5; newer combinations are not claimed until validated. Runtime imports also need to be available on executors. Missing dependencies, missing columns, missing ranking groups, unsupported distributed validation settings, and invalid native parameters fail with errors from the bridge or LightGBM.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m build
python -m twine check dist/*
```

See [CHANGELOG.md](CHANGELOG.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
