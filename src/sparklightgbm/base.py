from ._execution import train_native
from ._validation import check_columns, require_runtime

class BaseLightGBM:
    kind = "regressor"
    def __init__(self, features_col="features", label_col="label", prediction_col="prediction", raw_prediction_col="rawPrediction", probability_col="probability", leaf_prediction_col="leafPrediction", weight_col=None, group_col=None, categorical_feature=None, validation_data=None, early_stopping_rounds=None, seed=0, num_workers=None, local_listen_port=12400, prediction_batch_size=1024, objective=None, **params):
        if num_workers is not None and (isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1):
            raise ValueError("num_workers must be None or a positive integer")
        if isinstance(prediction_batch_size, bool) or not isinstance(prediction_batch_size, int) or prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be a positive integer")
        self.features_col, self.label_col = features_col, label_col; self.prediction_col, self.raw_prediction_col = prediction_col, raw_prediction_col; self.probability_col, self.leaf_prediction_col = probability_col, leaf_prediction_col
        self.weight_col, self.group_col, self.categorical_feature = weight_col, group_col, categorical_feature; self.validation_data, self.early_stopping_rounds = validation_data, early_stopping_rounds; self.seed, self.num_workers, self.local_listen_port, self.prediction_batch_size, self.objective, self.params = seed, num_workers, local_listen_port, prediction_batch_size, objective, params
    def fit(self, dataset, params=None, validation_data=None):
        require_runtime(); check_columns(dataset, self.features_col, self.label_col)
        if params: self.params.update(params)
        from .models import model_for_kind
        valid = validation_data if validation_data is not None else self.validation_data
        booster, n_features, classes = train_native(self, dataset, valid)
        return model_for_kind(self.kind)(booster, self, n_features, classes)
