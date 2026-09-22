# Architecture audit

## Execution paths

Single-worker training intentionally provides a compatibility fallback:

```text
Spark DataFrame -> partition rows -> driver NumPy arrays -> LightGBM Dataset -> Booster
```

This path collects training and validation rows and is suitable only when they fit in driver memory.

Distributed classifier and regressor training follows:

```text
Spark DataFrame -> repartitioned barrier stage -> executor-local contiguous NumPy arrays
                -> executor-local LightGBM Datasets -> native LightGBM network
                -> distributed tree construction -> model string metadata to driver
```

Training rows are never collected by the driver. The driver collects only bounded metadata (class labels), one serialized model per worker, and barrier results. Validation rows are tagged, co-partitioned with training shards, and remain executor-local. Native shard metrics that are mathematically decomposable are combined through the Spark barrier before early stopping, ensuring every worker stops on the same iteration.

Distributed ranking remains disabled because preserving complete query groups per worker needs a dedicated partitioning contract.

## Data movement and memory

The distributed path does not use pandas. Spark rows are decoded once into feature, label, and optional-weight buffers, then converted into contiguous NumPy matrices. One training and, when configured, one validation matrix are resident per worker. The current Python-row boundary remains a measurable cost and is a benchmark target; replacing it requires evidence that Arrow or another optional path improves end-to-end behavior without reducing portability.

Inference uses `RDD.mapPartitions`, constructs or reuses one Booster per Python worker, batches feature vectors into contiguous matrices, and calls native prediction once per requested mode and batch. Null feature vectors produce null outputs consistently.

## Known constraints

- Barrier jobs require all worker slots concurrently and stable executor networking.
- Exact distributed early stopping is unavailable for non-decomposable shard metrics such as AUC and average precision.
- Worker loss aborts native distributed training; transparent recovery would require a full coordinated restart.
- Spark ML `Estimator`, `Model`, and persistence interfaces are not implemented yet.
- Model artifacts are native LightGBM files plus lightweight metadata, not `PipelineModel` artifacts.

These constraints are kept explicit instead of adding silent fallbacks that collect distributed data to the driver.
