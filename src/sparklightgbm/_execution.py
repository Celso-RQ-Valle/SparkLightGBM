"""Spark-to-native execution. No model algorithm is implemented here."""
from __future__ import annotations
import numpy as np
from .errors import SparkLightGBMConfigurationError, SparkLightGBMNetworkError, SparkLightGBMWorkerError
from ._validation import check_columns, require_runtime

def _vector(value):
    if value is None: return None
    if hasattr(value, "toArray"): return value.toArray()
    return np.asarray(value, dtype=float)

def collect_training_data(df, features_col, label_col, weight_col=None, group_col=None):
    check_columns(df, features_col, label_col)
    columns = [features_col, label_col] + ([weight_col] if weight_col else []) + ([group_col] if group_col else [])
    def rows(partition):
        for row in partition:
            vals = list(row); x = _vector(vals[0])
            if x is None:
                continue
            offset = 2
            weight = float(vals[offset]) if weight_col else None
            offset += 1 if weight_col else 0
            group = int(vals[offset]) if group_col else None
            yield (x, float(vals[1]), weight, group)
    parts = df.select(*columns).rdd.mapPartitions(rows).collect()
    if not parts: raise ValueError("The training DataFrame contains no usable feature rows")
    x = np.asarray([p[0] for p in parts], dtype=float); y = np.asarray([p[1] for p in parts], dtype=float)
    w = np.asarray([p[2] for p in parts], dtype=float) if weight_col else None
    return x, y, w, ([p[3] for p in parts] if group_col else None)

def resolve_num_workers(estimator, df, validation_data=None):
    """Choose conservative Spark parallelism when no override is supplied.

    Automatic cluster execution reserves one Spark slot and uses at most four
    workers. The input partition count remains an upper bound. Local mode and
    ranking use the single-worker compatibility path.
    """
    if estimator.num_workers is not None:
        return estimator.num_workers
    # Local mode is faster without barrier networking. Distributed ranking is
    # deferred until groups can be kept intact across worker boundaries.
    if estimator.kind == "ranker":
        return 1
    sc = df.sparkSession.sparkContext
    if sc.master.lower().startswith("local"):
        return 1
    available_slots = max(1, int(sc.defaultParallelism))
    input_partitions = max(1, int(df.rdd.getNumPartitions()))
    usable_slots = max(1, available_slots - 1)
    return min(4, usable_slots, input_partitions)

def train_native(estimator, df, validation_data=None):
    _, lgb = require_runtime()
    if estimator.early_stopping_rounds and validation_data is None:
        raise SparkLightGBMConfigurationError("early_stopping_rounds requires validation_data")
    num_workers = resolve_num_workers(estimator, df, validation_data)
    if num_workers > 1:
        return train_native_distributed(estimator, df, validation_data, lgb, num_workers)
    x, y, w, groups = collect_training_data(df, estimator.features_col, estimator.label_col, estimator.weight_col, estimator.group_col)
    params = dict(estimator.params); params.setdefault("device_type", "cpu"); params.setdefault("seed", estimator.seed); params.setdefault("verbosity", -1)
    num_boost_round = int(params.pop("n_estimators", params.pop("num_boost_round", 100)))
    if estimator.kind == "classifier":
        inferred_classes = len(np.unique(y))
        objective = estimator.objective or ("multiclass" if (estimator.num_class or inferred_classes) > 2 else "binary")
        params.setdefault("objective", objective)
        if objective == "multiclass": params.setdefault("num_class", estimator.num_class or inferred_classes)
    elif estimator.kind == "ranker":
        params.setdefault("objective", estimator.objective or "lambdarank")
    else:
        params.setdefault("objective", estimator.objective or "regression")
    train_group = None
    if estimator.kind == "ranker":
        if not groups: raise ValueError("LightGBMRanker requires group_col")
        order = np.argsort(np.asarray(groups)); x, y = x[order], y[order]
        if w is not None: w = w[order]
        _, counts = np.unique(np.asarray(groups)[order], return_counts=True); train_group = counts.tolist()
    categorical = estimator.categorical_feature or "auto"
    train_set = lgb.Dataset(x, label=y, weight=w, group=train_group, categorical_feature=categorical, free_raw_data=False)
    valid_sets = [train_set]
    valid_names = ["training"]
    callbacks = []
    if validation_data is not None:
        vx, vy, vw, vg = collect_training_data(validation_data, estimator.features_col, estimator.label_col, estimator.weight_col, estimator.group_col)
        valid_group = None
        if estimator.kind == "ranker":
            if not vg: raise ValueError("Validation data for LightGBMRanker requires group_col")
            order = np.argsort(np.asarray(vg)); vx, vy = vx[order], vy[order]
            if vw is not None: vw = vw[order]
            _, vc = np.unique(np.asarray(vg)[order], return_counts=True); valid_group = vc.tolist()
        valid_sets.append(lgb.Dataset(vx, label=vy, weight=vw, group=valid_group, reference=train_set, categorical_feature=categorical))
        valid_names.append("validation")
        if estimator.early_stopping_rounds: callbacks.append(lgb.early_stopping(estimator.early_stopping_rounds, verbose=False))
    booster = lgb.train(params, train_set, num_boost_round=num_boost_round, valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks)
    classes = np.unique(y).tolist() if estimator.kind == "classifier" else None
    return booster, x.shape[1], classes

def _distributed_input(df, estimator, num_workers, is_validation):
    """Assign matching train/validation shards without collecting their rows."""
    from pyspark.sql import functions as F
    columns = [F.col(estimator.features_col).alias("__slgbm_features"), F.col(estimator.label_col).alias("__slgbm_label")]
    columns.append(F.col(estimator.weight_col).alias("__slgbm_weight") if estimator.weight_col else F.lit(None).cast("double").alias("__slgbm_weight"))
    return (df.select(*columns)
            .repartition(num_workers)
            .withColumn("__slgbm_worker", F.spark_partition_id())
            .withColumn("__slgbm_validation", F.lit(is_validation)))


def _distributed_metric_names(params):
    metric = params.get("metric")
    if metric is None:
        return []
    if isinstance(metric, str):
        return [name.strip().lower() for name in metric.split(",")]
    return [str(name).strip().lower() for name in metric]


def _worker_host(allow_loopback):
    """Return an address LightGBM can match to the current machine."""
    import os
    import socket

    configured = os.environ.get("SPARK_LOCAL_IP")
    # LightGBM may not identify 127/8 as a local machine on Windows. In local
    # Spark mode, prefer a hostname-derived interface address when available.
    if configured and not (allow_loopback and configured.startswith("127.")):
        return configured
    for name in (socket.getfqdn(), socket.gethostname()):
        try:
            resolved = socket.gethostbyname(name)
        except socket.gaierror:
            continue
        if resolved and (allow_loopback or not resolved.startswith("127.")):
            return resolved
    return configured or "127.0.0.1"


def _synchronize_evaluation(context, train_denominator, valid_denominator):
    """Aggregate decomposable validation metrics before early-stopping runs."""
    import json

    def callback(env):
        if not env.evaluation_result_list:
            return
        payload = {
            "train": train_denominator,
            "valid": valid_denominator,
            "values": [float(item[2]) for item in env.evaluation_result_list],
        }
        gathered = [json.loads(item) for item in context.allGather(json.dumps(payload))]
        synchronized = []
        for index, item in enumerate(env.evaluation_result_list):
            denominator_key = "train" if item[0] == "training" else "valid"
            denominator = sum(peer[denominator_key] for peer in gathered)
            value = sum(peer["values"][index] * peer[denominator_key] for peer in gathered) / denominator
            values = (item[0], item[1], value, *item[3:])
            synchronized.append(type(item)(*values) if hasattr(item, "dataset_name") else values)
        env.evaluation_result_list[:] = synchronized

    callback.order = 25
    callback.before_iteration = False
    return callback


def train_native_distributed(estimator, df, validation_data, lgb, num_workers):
    """Coordinate official LightGBM's data-parallel learner from Spark barriers."""
    import json
    from pyspark import BarrierTaskContext
    check_columns(df, estimator.features_col, estimator.label_col)
    if validation_data is not None:
        check_columns(validation_data, estimator.features_col, estimator.label_col)
    if estimator.kind == "ranker":
        raise SparkLightGBMConfigurationError("Distributed ranking is not supported until ranking groups can be kept worker-local")
    if df.limit(num_workers).count() < num_workers:
        raise SparkLightGBMConfigurationError(f"num_workers={num_workers} requires at least one training row per worker")
    if validation_data is not None and validation_data.limit(num_workers).count() < num_workers:
        raise SparkLightGBMConfigurationError(f"num_workers={num_workers} requires at least one validation row per worker")

    classes = None
    if estimator.kind == "classifier":
        classes = sorted(float(row[0]) for row in df.select(estimator.label_col).distinct().collect())
    objective = estimator.objective
    if objective is None:
        objective = "multiclass" if estimator.kind == "classifier" and (estimator.num_class or len(classes)) > 2 else "binary" if estimator.kind == "classifier" else "regression"

    train_df = _distributed_input(df, estimator, num_workers, False)
    if validation_data is not None:
        combined = train_df.unionByName(_distributed_input(validation_data, estimator, num_workers, True))
    else:
        combined = train_df
    worker_df = combined.repartitionByRange(num_workers, "__slgbm_worker").select("__slgbm_features", "__slgbm_label", "__slgbm_weight", "__slgbm_validation")
    allow_loopback = df.sparkSession.sparkContext.master.lower().startswith("local")
    has_validation = validation_data is not None
    has_weight = estimator.weight_col is not None
    native_params = dict(estimator.params)
    seed = estimator.seed
    local_listen_port = estimator.local_listen_port
    categorical_feature = estimator.categorical_feature or "auto"
    estimator_kind = estimator.kind
    configured_num_class = getattr(estimator, "num_class", None)
    early_stopping_rounds = estimator.early_stopping_rounds
    non_decomposable_metrics = {"auc", "average_precision", "map", "ndcg"}
    requested_metrics = set(_distributed_metric_names(native_params))
    unsupported_metrics = requested_metrics & non_decomposable_metrics
    if has_validation and early_stopping_rounds and unsupported_metrics:
        names = ", ".join(sorted(unsupported_metrics))
        raise SparkLightGBMConfigurationError(f"Distributed early stopping cannot exactly aggregate metric(s): {names}")

    def worker(rows):
        context = BarrierTaskContext.get()
        train_x, train_y, train_weight = [], [], []
        valid_x, valid_y, valid_weight = [], [], []
        for row in rows:
            value = _vector(row[0])
            if value is None:
                continue
            if row[3]:
                valid_x.append(value); valid_y.append(float(row[1])); valid_weight.append(float(row[2]) if row[2] is not None else None)
            else:
                train_x.append(value); train_y.append(float(row[1])); train_weight.append(float(row[2]) if row[2] is not None else None)
        if not train_x:
            raise SparkLightGBMWorkerError(f"LightGBM worker {context.partitionId()} received no training rows")
        if has_validation and not valid_x:
            raise SparkLightGBMWorkerError(f"LightGBM worker {context.partitionId()} received no validation rows")
        host = _worker_host(allow_loopback)
        if host.startswith("127.") and not allow_loopback:
            raise SparkLightGBMNetworkError("A distributed worker resolved to a loopback address; configure SPARK_LOCAL_IP with an executor-reachable address")
        port = local_listen_port + context.partitionId()
        peers = context.allGather(json.dumps({"host": host, "port": port}))
        x = np.ascontiguousarray(train_x, dtype=float); y = np.asarray(train_y, dtype=float)
        weight = np.asarray(train_weight, dtype=float) if has_weight else None
        params = dict(native_params); rounds = int(params.pop("n_estimators", params.pop("num_boost_round", 100)))
        params.setdefault("device_type", "cpu"); params.setdefault("seed", seed); params.setdefault("verbosity", -1)
        # Each barrier task occupies one Spark CPU slot. Restrict its native
        # thread pool unless the user explicitly requests a different value.
        params.setdefault("num_threads", 1)
        params.setdefault("objective", objective)
        if estimator_kind == "classifier" and objective == "multiclass": params.setdefault("num_class", configured_num_class or len(classes))
        params.update({"tree_learner": "data", "num_machines": num_workers, "machines": ",".join(f"{json.loads(p)['host']}:{json.loads(p)['port']}" for p in peers), "local_listen_port": port})
        train_set = lgb.Dataset(x, label=y, weight=weight, categorical_feature=categorical_feature, free_raw_data=False)
        valid_sets = [train_set]
        valid_names = ["training"]
        callbacks = []
        train_denominator = float(np.sum(weight)) if weight is not None else float(len(y))
        if has_validation:
            vx = np.ascontiguousarray(valid_x, dtype=float); vy = np.asarray(valid_y, dtype=float)
            vw = np.asarray(valid_weight, dtype=float) if has_weight else None
            valid_sets.append(lgb.Dataset(vx, label=vy, weight=vw, reference=train_set, categorical_feature=categorical_feature, free_raw_data=False))
            valid_names.append("validation")
            valid_denominator = float(np.sum(vw)) if vw is not None else float(len(vy))
            callbacks.append(_synchronize_evaluation(context, train_denominator, valid_denominator))
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        try:
            booster = lgb.train(params, train_set, num_boost_round=rounds, valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks)
        except Exception as exc:
            raise SparkLightGBMWorkerError(f"Native LightGBM training failed on worker {context.partitionId()}: {exc}") from exc
        rank = context.partitionId()
        yield json.dumps({"rank": rank, "model": booster.model_to_string() if rank == 0 else None})
    models = worker_df.rdd.barrier().mapPartitions(worker).collect()
    primary = next(json.loads(item)["model"] for item in models if json.loads(item)["rank"] == 0)
    booster = lgb.Booster(model_str=primary)
    return booster, booster.num_feature(), classes
