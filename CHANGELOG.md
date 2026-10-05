# Changelog

## 0.9.2 - 2026-10-05

- Allow Python 3.8 and newer without a speculative upper version cap.
- Allow PySpark 4.x in dependency extras and runtime validation, retaining the PySpark 3.4 minimum.
- Expand the CI integration matrix across Python 3.8-3.14 and Spark 3.4, 3.5, 4.0, 4.1, and 4.2.
- Document compatible runtime combinations and the requirement for classic Spark with RDD access.

## 0.9.1 - 2026-10-04

- Default distributed training and Spark prediction to native LightGBM/OpenMP automatic threading.
- Honor explicit thread limits and native aliases during Spark prediction, including overrides when loading native models.
- Document thread selection and Spark CPU allocation considerations.
- Add regression tests for prediction threading and distributed training overrides.

## 0.9.0 - 2026-09-22

- Prepare metadata, documentation, source distribution, wheel, and CI for a public PyPI release.
- Define the supported Python 3.9-3.12, PySpark 3.4-3.5 compatibility policy.
- Make `num_workers=None` the documented default and conservatively cap automatic cluster training at four workers while reserving one reported Spark slot.
- Keep validation shards distributed during native multi-worker training.
- Synchronize decomposable validation metrics across barrier workers for consistent distributed early stopping.
- Batch inference by Spark partition and reuse native boosters in Python workers.
- Make distributed objective inference globally consistent and reduce model artifacts returned to the driver.
- Add library-specific configuration, network, and worker errors plus a reproducible benchmark harness.

## 0.1.0b1 - 2026-09-20

Beta release for objective runtime testing. Includes Spark DataFrame estimators for classification, regression, quantile regression, and LambdaRank; native predictions, SHAP contributions, feature importance, early stopping in single-worker mode, categorical values, weights, missing values, seeds, native model persistence, and experimental native distributed training through Spark barriers.
