# SparkLightGBM

SparkLightGBM is a Python bridge between Apache Spark DataFrames and the official [LightGBM](https://github.com/lightgbm-org/LightGBM) Python package. It provides Spark-facing estimators for binary and multiclass classification, regression, quantile regression, and LambdaRank without requiring SynapseML or a Scala/JVM extension.

## Status

`0.1.0b1` is a Beta release for objective runtime testing. It is not production-stable. The future stable API and release target is `1.0.0`; public APIs may still change before then.

## Installation

```bash
pip install sparklightgbm pyspark
```

The package requires Python 3.9+, NumPy, and official LightGBM 4.x. Spark is used at runtime and should be installed on the driver and every executor. The optional development installation is:

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
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    seed=7,
)

model = estimator.fit(train_df)
predictions = model.transform(test_df)
predictions.select("prediction", "probability").show()
```

## Estimators and input parameters

All three estimators accept the shared parameters below. Any additional keyword argument is forwarded to native LightGBM as a parameter. This allows new LightGBM parameters to be used without waiting for a SparkLightGBM wrapper release.

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
| `early_stopping_rounds` | `int \| None` | `None` | Rounds without validation improvement; requires validation data and `num_workers=1`. |
| `seed` | `int` | `0` | Seed passed to LightGBM. |
| `num_workers` | `int` | `1` | `1` uses driver training; values above `1` enable native data-parallel Spark barrier training. |
| `local_listen_port` | `int` | `12400` | Distributed listener base port; worker `n` uses `local_listen_port + n`. |
| `objective` | `str \| None` | `None` | Native objective: defaults to `binary`/`multiclass`, `regression`, or `lambdarank`; `quantile` is supported for regression. |
| `**params` | `str`/`int`/`float`/`bool`/native value | - | Additional native LightGBM parameters, including `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `max_bin`, `bagging_seed`, and `verbosity`. |

Estimator-specific parameters:

| Estimator | Parameter | Accepted type / default | Description |
| --- | --- | --- | --- |
| `LightGBMClassifier` | `num_class` | `int \| None`, default `None` | Number of classes; if omitted, inferred from labels. |
| `LightGBMRegressor` | - | - | Supports native regression parameters and `objective="quantile"`; set native `alpha` for the target quantile. |
| `LightGBMRanker` | - | - | Uses LambdaRank by default. `group_col` identifies groups; rows are sorted by group before native training. |

`n_estimators` is accepted as a LightGBM parameter and controls boosting rounds. `fit(params={...})` can supply or override native parameters:

```python
from sparklightgbm import LightGBMRegressor

model = LightGBMRegressor(features_col="features", label_col="target").fit(
    train_df,
    params={"objective": "quantile", "alpha": 0.9, "n_estimators": 200, "learning_rate": 0.03},
)
```

For the complete version-specific native parameter set, see the [official LightGBM Parameters documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html). SparkLightGBM passes native parameters to `lightgbm.Dataset` and `lightgbm.train`; it does not use the scikit-learn wrapper.

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

Validation data and early stopping are currently supported with `num_workers=1`. Distributed native mode rejects them explicitly because validation coordination across native workers is not implemented in this beta.

## Distributed execution and limitations

With default `num_workers=1`, Spark partitions are converted to NumPy arrays and collected to the driver before native training. This mode is intended for local development and smaller datasets whose feature matrix fits in driver memory.

With `num_workers > 1`, Spark repartitions the input, starts one barrier task per worker, exchanges worker addresses, and coordinates LightGBM's native data-parallel learner. Distributed execution requires:

- SparkLightGBM, NumPy, and the same compatible LightGBM build on every executor.
- Spark barrier execution support.
- Executor hostnames or IP addresses resolvable and reachable from every other executor.
- Listener ports available between executors; set `local_listen_port` if `12400 + worker_id` is unavailable.
- Correct executor networking. `SPARK_LOCAL_IP` can provide the advertised worker address when hostname resolution is unsuitable.

Distributed mode currently does not support `validation_data` or early stopping. Ranking groups should be partitioned so rows from the same group are not split across workers. Test distributed behavior on the target local Spark, Databricks, or standard cluster before relying on it.

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

The native model file is portable across Spark jobs with a compatible LightGBM installation. It is not a Spark ML `PipelineModel` format in this beta.

`save_native_model(path)` accepts a filesystem path (`str` or `pathlib.Path`). `LightGBMClassificationModel.load_native_model(path, **kwargs)`, `LightGBMRegressionModel.load_native_model(path, **kwargs)`, and `LightGBMRankingModel.load_native_model(path, **kwargs)` accept the native model path plus optional `features_col`, `prediction_col`, `raw_prediction_col`, `probability_col`, `leaf_prediction_col`, and `kind` strings used to reconstruct Spark output behavior.

## Compatibility and errors

SparkLightGBM validates required columns and checks that PySpark 3.3+ and LightGBM are available on the driver. Runtime imports also need to be available on executors. Missing dependencies, missing columns, missing ranking groups, unsupported distributed validation settings, and invalid native parameters fail with errors from the bridge or LightGBM.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m build
python -m twine check dist/*
```

See [CHANGELOG.md](CHANGELOG.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
