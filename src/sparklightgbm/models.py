from .explainability import feature_importance, shap_values
from .persistence import load_native_model, save_native_model

_BOOSTER_CACHE = {}


def _cached_booster(model_string):
    import hashlib
    import lightgbm as lgb
    key = hashlib.sha256(model_string.encode("utf-8")).digest()
    booster = _BOOSTER_CACHE.get(key)
    if booster is None:
        if len(_BOOSTER_CACHE) >= 4:
            _BOOSTER_CACHE.clear()
        booster = lgb.Booster(model_str=model_string)
        _BOOSTER_CACHE[key] = booster
    return booster


def _batched(iterator, size):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _score_partition(rows, model_string, feature_index, base_indexes, batch_size, kind, binary, include_raw, include_probability, include_leaf):
    import numpy as np
    booster = _cached_booster(model_string)
    for rows_batch in _batched(rows, batch_size):
        features = []
        valid_positions = []
        outputs = [[None, None if include_raw else False, None if include_probability else False, None if include_leaf else False] for _ in rows_batch]
        for position, row in enumerate(rows_batch):
            value = row[feature_index]
            if value is None:
                continue
            if hasattr(value, "toArray"):
                value = value.toArray()
            features.append(value)
            valid_positions.append(position)
        if features:
            matrix = np.ascontiguousarray(features, dtype=float)
            if kind == "classifier":
                native_raw = np.asarray(booster.predict(matrix, raw_score=True, num_threads=1), dtype=float)
                if binary:
                    scores = native_raw.reshape(-1)
                    raw = np.column_stack((-scores, scores))
                    positive = np.empty_like(scores)
                    nonnegative = scores >= 0
                    positive[nonnegative] = 1.0 / (1.0 + np.exp(-scores[nonnegative]))
                    exp_scores = np.exp(scores[~nonnegative])
                    positive[~nonnegative] = exp_scores / (1.0 + exp_scores)
                    probability = np.column_stack((1.0 - positive, positive))
                else:
                    raw = native_raw.reshape(len(valid_positions), -1)
                    shifted = raw - np.max(raw, axis=1, keepdims=True)
                    probability = np.exp(shifted)
                    probability /= probability.sum(axis=1, keepdims=True)
                prediction = np.argmax(raw, axis=1).astype(float)
            else:
                prediction = np.asarray(booster.predict(matrix, num_threads=1), dtype=float).reshape(-1)
                raw = np.asarray(booster.predict(matrix, raw_score=True, num_threads=1), dtype=float).reshape(-1) if include_raw else None
                probability = None
            leaves = np.asarray(booster.predict(matrix, pred_leaf=True, num_threads=1)).reshape(len(valid_positions), -1) if include_leaf else None
            for batch_index, position in enumerate(valid_positions):
                outputs[position][0] = float(prediction[batch_index])
                if include_raw:
                    outputs[position][1] = raw[batch_index].astype(float).tolist() if kind == "classifier" else float(raw[batch_index])
                if include_probability:
                    outputs[position][2] = probability[batch_index].astype(float).tolist()
                if include_leaf:
                    outputs[position][3] = leaves[batch_index].astype(int).tolist()
        for row, values in zip(rows_batch, outputs):
            appended = [values[0]]
            if include_raw:
                appended.append(values[1])
            if include_probability:
                appended.append(values[2])
            if include_leaf:
                appended.append(values[3])
            yield tuple(row[index] for index in base_indexes) + tuple(appended)

class BaseLightGBMModel:
    def __init__(self, booster, estimator, n_features, classes=None): self.booster, self.estimator, self.n_features, self.classes_ = booster, estimator, n_features, classes
    def _is_binary_classifier(self):
        return self.estimator.kind == "classifier" and self.booster.num_model_per_iteration() == 1
    @staticmethod
    def _sigmoid(score):
        """Numerically stable sigmoid used by LightGBM's binary objective."""
        import math
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-score))
        exp_score = math.exp(score)
        return exp_score / (1.0 + exp_score)
    def _prediction_from_raw(self, raw):
        import numpy as np
        if self.estimator.kind != "classifier":
            return float(raw)
        scores = np.asarray(raw, dtype=float).reshape(-1)
        return float(np.argmax(scores))
    def _probability_from_raw(self, raw):
        import numpy as np
        scores = np.asarray(raw, dtype=float).reshape(-1)
        if self._is_binary_classifier():
            positive = self._sigmoid(float(scores[-1]))
            return [1.0 - positive, positive]
        # LightGBM's multiclass objective converts margins with softmax.
        shifted = scores - np.max(scores)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        return probabilities.tolist()
    def _predict(self, value, kind):
        import numpy as np
        if value is None:
            return None
        if hasattr(value, "toArray"): value = value.toArray()
        binary = self._is_binary_classifier()
        # Classification prediction and probability are deliberately derived from
        # the same raw margins exposed in rawPrediction, as Spark probabilistic
        # classifiers (and SynapseML) require.
        raw_based = kind in {"raw", "prob", "prediction"}
        result = self.booster.predict(np.asarray([value], dtype=float), pred_leaf=(kind == "leaf"), raw_score=raw_based, pred_contrib=(kind == "shap"))[0]
        if kind == "raw" and binary:
            score = float(np.asarray(result).reshape(-1)[0])
            return [-score, score]
        if kind == "prediction":
            raw = [-float(result), float(result)] if binary else result
            return self._prediction_from_raw(raw)
        if kind == "prob":
            raw = [-float(result), float(result)] if binary else result
            return self._probability_from_raw(raw)
        if kind == "leaf":
            return [int(value) for value in np.asarray(result).reshape(-1).tolist()]
        if hasattr(result, "tolist"):
            return result.tolist()
        if isinstance(result, list):
            return [float(value) for value in result]
        return float(result)
    def transform(self, dataset, prediction_col=None, raw_prediction_col=None, probability_col=None, leaf_prediction_col=None):
        from pyspark.sql.types import ArrayType, DoubleType, LongType, StructField, StructType
        pcol = prediction_col or self.estimator.prediction_col
        probability = self.estimator.kind == "classifier"
        raw_name = raw_prediction_col or self.estimator.raw_prediction_col
        prob_name = (probability_col or self.estimator.probability_col) if probability else None
        leaf_name = leaf_prediction_col or self.estimator.leaf_prediction_col
        output_names = [pcol] + ([raw_name] if raw_name else []) + ([prob_name] if prob_name else []) + ([leaf_name] if leaf_name else [])
        if len(set(output_names)) != len(output_names):
            raise ValueError("Prediction output column names must be unique")
        feature_index = dataset.columns.index(self.estimator.features_col)
        base_indexes = [index for index, field in enumerate(dataset.schema.fields) if field.name not in output_names]
        fields = [dataset.schema.fields[index] for index in base_indexes]
        fields.append(StructField(pcol, DoubleType(), True))
        if raw_name:
            fields.append(StructField(raw_name, ArrayType(DoubleType(), containsNull=False) if probability else DoubleType(), True))
        if prob_name:
            fields.append(StructField(prob_name, ArrayType(DoubleType(), containsNull=False), True))
        if leaf_name:
            fields.append(StructField(leaf_name, ArrayType(LongType(), containsNull=False), True))
        model_string = self.booster.model_to_string()
        batch_size = getattr(self.estimator, "prediction_batch_size", 1024)
        kind = self.estimator.kind
        binary = self._is_binary_classifier()
        scored = dataset.rdd.mapPartitions(lambda rows: _score_partition(rows, model_string, feature_index, base_indexes, batch_size, kind, binary, bool(raw_name), bool(prob_name), bool(leaf_name)))
        return dataset.sparkSession.createDataFrame(scored, StructType(fields))
    def predict_raw(self, matrix):
        result = self.booster.predict(matrix, raw_score=True)
        if not self._is_binary_classifier():
            return result
        import numpy as np
        scores = np.asarray(result, dtype=float).reshape(-1)
        return np.column_stack((-scores, scores))
    def predict_probability(self, matrix):
        if not self._is_binary_classifier():
            return self.booster.predict(matrix)
        import numpy as np
        scores = np.asarray(self.booster.predict(matrix, raw_score=True), dtype=float).reshape(-1)
        positive = np.asarray([self._sigmoid(score) for score in scores])
        return np.column_stack((1.0 - positive, positive))
    def predict_leaf(self, matrix): return self.booster.predict(matrix, pred_leaf=True)
    def predict_shap(self, matrix): return shap_values(self.booster, matrix)
    def feature_importance(self, importance_type="split"): return feature_importance(self.booster, importance_type)
    def save_native_model(self, path): save_native_model(self.booster, path, {"kind": self.estimator.kind, "n_features": self.n_features})
    @classmethod
    def load_native_model(cls, path, **kwargs):
        booster = load_native_model(path)
        class E: pass
        default_kind = "classifier" if cls.__name__.startswith("LightGBMClassification") else "ranker" if cls.__name__.startswith("LightGBMRanking") else "regressor"
        e = E(); e.kind = kwargs.pop("kind", default_kind); e.features_col = kwargs.pop("features_col", "features"); e.prediction_col = kwargs.pop("prediction_col", "prediction"); e.raw_prediction_col = kwargs.pop("raw_prediction_col", "rawPrediction"); e.probability_col = kwargs.pop("probability_col", "probability"); e.leaf_prediction_col = kwargs.pop("leaf_prediction_col", "leafPrediction"); e.prediction_batch_size = kwargs.pop("prediction_batch_size", 1024)
        return cls(booster, e, booster.num_feature())

class LightGBMClassificationModel(BaseLightGBMModel): pass
class LightGBMRegressionModel(BaseLightGBMModel): pass
class LightGBMRankingModel(BaseLightGBMModel): pass
def model_for_kind(kind): return {"classifier": LightGBMClassificationModel, "regressor": LightGBMRegressionModel, "ranker": LightGBMRankingModel}[kind]
