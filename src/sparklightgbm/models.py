from .explainability import feature_importance, shap_values
from .persistence import load_native_model, save_native_model

class BaseLightGBMModel:
    def __init__(self, booster, estimator, n_features, classes=None): self.booster, self.estimator, self.n_features, self.classes_ = booster, estimator, n_features, classes
    def _predict(self, value, kind):
        import numpy as np
        if hasattr(value, "toArray"): value = value.toArray()
        result = self.booster.predict(np.asarray([value], dtype=float), pred_leaf=(kind == "leaf"), raw_score=(kind == "raw"), pred_contrib=(kind == "shap"))[0]
        if kind == "prediction" and self.estimator.kind == "classifier":
            result = int(np.argmax(result)) if hasattr(result, "__len__") else int(result >= 0.5)
        if kind == "prob":
            return [float(value) for value in np.asarray(result).reshape(-1).tolist()]
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
            raw_type = ArrayType(DoubleType()) if probability and getattr(self.booster, "num_model_per_iteration", lambda: 1)() > 1 else DoubleType()
            out = out.withColumn(raw_name, udf(lambda x: self._predict(x, "raw"), raw_type)(self.estimator.features_col))
        if probability:
            prob_name = probability_col or self.estimator.probability_col
            if prob_name: out = out.withColumn(prob_name, udf(lambda x: self._predict(x, "prob"), ArrayType(DoubleType(), containsNull=False))(self.estimator.features_col))
        leaf_name = leaf_prediction_col or self.estimator.leaf_prediction_col
        if leaf_name: out = out.withColumn(leaf_name, udf(lambda x: self._predict(x, "leaf"), ArrayType(LongType(), containsNull=False))(self.estimator.features_col))
        return out
    def predict_raw(self, matrix): return self.booster.predict(matrix, raw_score=True)
    def predict_probability(self, matrix): return self.booster.predict(matrix)
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
