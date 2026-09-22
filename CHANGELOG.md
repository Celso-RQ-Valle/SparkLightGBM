# Changelog

## Unreleased

- Keep validation shards distributed during native multi-worker training.
- Synchronize decomposable validation metrics across barrier workers for consistent distributed early stopping.
- Batch inference by Spark partition and reuse native boosters in Python workers.
- Make distributed objective inference globally consistent and reduce model artifacts returned to the driver.
- Add library-specific configuration, network, and worker errors plus a reproducible benchmark harness.

## 0.1.0b1 - 2026-09-20

Beta release for objective runtime testing. Includes Spark DataFrame estimators for classification, regression, quantile regression, and LambdaRank; native predictions, SHAP contributions, feature importance, early stopping in single-worker mode, categorical values, weights, missing values, seeds, native model persistence, and experimental native distributed training through Spark barriers.
