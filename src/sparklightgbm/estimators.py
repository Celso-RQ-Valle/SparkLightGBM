from .base import BaseLightGBM
class LightGBMClassifier(BaseLightGBM):
    kind = "classifier"
    def __init__(self, num_class=None, **kwargs): self.num_class = num_class; super().__init__(**kwargs)
class LightGBMRegressor(BaseLightGBM):
    kind = "regressor"
class LightGBMRanker(BaseLightGBM):
    kind = "ranker"
