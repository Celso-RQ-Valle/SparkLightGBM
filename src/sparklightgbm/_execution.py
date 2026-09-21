"""Spark-to-native execution. No model algorithm is implemented here."""
from __future__ import annotations
import numpy as np
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
            if x is not None: yield (x, float(vals[1]), float(vals[2]) if weight_col else None, int(vals[3]) if group_col else None)
    parts = df.select(*columns).rdd.mapPartitions(rows).collect()
    if not parts: raise ValueError("The training DataFrame contains no usable feature rows")
    x = np.asarray([p[0] for p in parts], dtype=float); y = np.asarray([p[1] for p in parts], dtype=float)
    w = np.asarray([p[2] for p in parts], dtype=float) if weight_col else None
    return x, y, w, ([p[3] for p in parts] if group_col else None)

def train_native(estimator, df, validation_data=None):
    _, lgb = require_runtime()
    if estimator.num_workers > 1:
        return train_native_distributed(estimator, df, lgb)
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

def train_native_distributed(estimator, df, lgb):
    """Coordinate official LightGBM's data-parallel learner from Spark barriers."""
    if estimator.early_stopping_rounds or estimator.validation_data:
        raise ValueError("validation_data and early stopping are currently supported with num_workers=1; use a validation split per worker for native distributed mode")
    import json
    import os
    import socket
    from pyspark import BarrierTaskContext
    columns = [estimator.features_col, estimator.label_col] + ([estimator.weight_col] if estimator.weight_col else [])
    worker_df = df.select(*columns).repartition(estimator.num_workers)
    def worker(rows):
        context = BarrierTaskContext.get()
        records = []
        for row in rows:
            vals = list(row); value = _vector(vals[0])
            if value is not None: records.append((value, float(vals[1]), float(vals[2]) if estimator.weight_col else None))
        if not records: raise ValueError("Every native LightGBM worker must receive at least one row")
        host = os.environ.get("SPARK_LOCAL_IP")
        if not host:
            try:
                host = socket.gethostbyname(socket.getfqdn())
            except socket.gaierror:
                host = socket.gethostbyname(socket.gethostname())
        if host.startswith("127.") and not socket.gethostname().lower() in {"localhost", "127.0.0.1"}:
            raise RuntimeError("LightGBM worker hostname resolved to loopback; configure executor DNS or SPARK_LOCAL_IP")
        port = estimator.local_listen_port + context.partitionId()
        peers = context.allGather(json.dumps({"host": host, "port": port}))
        x = np.asarray([r[0] for r in records], dtype=float); y = np.asarray([r[1] for r in records], dtype=float)
        weight = np.asarray([r[2] for r in records], dtype=float) if estimator.weight_col else None
        params = dict(estimator.params); rounds = int(params.pop("n_estimators", params.pop("num_boost_round", 100)))
        params.setdefault("device_type", "cpu"); params.setdefault("seed", estimator.seed); params.setdefault("verbosity", -1)
        params.setdefault("objective", estimator.objective or ("multiclass" if len(np.unique(y)) > 2 else "binary") if estimator.kind == "classifier" else "lambdarank" if estimator.kind == "ranker" else "regression")
        if estimator.kind == "classifier" and params["objective"] == "multiclass": params.setdefault("num_class", len(np.unique(y)))
        params.update({"tree_learner": "data", "num_machines": estimator.num_workers, "machines": " ".join(f"{json.loads(p)['host']}:{json.loads(p)['port']}" for p in peers), "local_listen_port": port})
        dataset = lgb.Dataset(x, label=y, weight=weight, categorical_feature=estimator.categorical_feature or "auto")
        booster = lgb.train(params, dataset, num_boost_round=rounds)
        yield json.dumps({"rank": context.partitionId(), "model": booster.model_to_string()})
    models = worker_df.rdd.barrier().mapPartitions(worker).collect()
    primary = next(json.loads(item)["model"] for item in models if json.loads(item)["rank"] == 0)
    booster = lgb.Booster(model_str=primary)
    return booster, booster.num_feature(), np.unique(df.select(estimator.label_col).rdd.map(lambda r: float(r[0])).collect()).tolist() if estimator.kind == "classifier" else None
