from .explainability import feature_importance, shap_values
from .persistence import load_native_model, save_native_model

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
        from pyspark.sql.functions import udf
        from pyspark.sql.types import ArrayType, DoubleType, LongType
        pcol = prediction_col or self.estimator.prediction_col
        probability = self.estimator.kind == "classifier"
        prediction_kind = "prediction" if probability else "value"
        out = dataset.withColumn(pcol, udf(lambda x: self._predict(x, prediction_kind), DoubleType())(self.estimator.features_col))
        raw_name = raw_prediction_col or self.estimator.raw_prediction_col
        if raw_name:
            raw_type = ArrayType(DoubleType(), containsNull=False) if probability else DoubleType()
            out = out.withColumn(raw_name, udf(lambda x: self._predict(x, "raw"), raw_type)(self.estimator.features_col))
        if probability:
            prob_name = probability_col or self.estimator.probability_col
            if prob_name: out = out.withColumn(prob_name, udf(lambda x: self._predict(x, "prob"), ArrayType(DoubleType(), containsNull=False))(self.estimator.features_col))
        leaf_name = leaf_prediction_col or self.estimator.leaf_prediction_col
        if leaf_name: out = out.withColumn(leaf_name, udf(lambda x: self._predict(x, "leaf"), ArrayType(LongType(), containsNull=False))(self.estimator.features_col))
        return out
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
        import math
        import numpy as np
        scores = np.asarray(self.booster.predict(matrix, raw_score=True), dtype=float).reshape(-1)
        return np.asarray([[1.0 / (1.0 + math.exp(score)), 1.0 / (1.0 + math.exp(-score))] for score in scores])
    def predict_leaf(self, matrix): return self.booster.predict(matrix, pred_leaf=True)
    def predict_shap(self, matrix): return shap_values(self.booster, matrix)
    def feature_importance(self, importance_type="split"): return feature_importance(self.booster, importance_type)
    def save_native_model(self, path): save_native_model(self.booster, path, {"kind": self.estimator.kind, "n_features": self.n_features})
    @classmethod
    def load_native_model(cls, path, **kwargs):
        booster = load_native_model(path)
        class E: pass
        default_kind = "classifier" if cls.__name__.startswith("LightGBMClassification") else "ranker" if cls.__name__.startswith("LightGBMRanking") else "regressor"
        e = E(); e.kind = kwargs.pop("kind", default_kind); e.features_col = kwargs.pop("features_col", "features"); e.prediction_col = kwargs.pop("prediction_col", "prediction"); e.raw_prediction_col = kwargs.pop("raw_prediction_col", "rawPrediction"); e.probability_col = kwargs.pop("probability_col", "probability"); e.leaf_prediction_col = kwargs.pop("leaf_prediction_col", "leafPrediction")
        return cls(booster, e, booster.num_feature())

class LightGBMClassificationModel(BaseLightGBMModel): pass
class LightGBMRegressionModel(BaseLightGBMModel): pass
class LightGBMRankingModel(BaseLightGBMModel): pass
def model_for_kind(kind): return {"classifier": LightGBMClassificationModel, "regressor": LightGBMRegressionModel, "ranker": LightGBMRankingModel}[kind]
