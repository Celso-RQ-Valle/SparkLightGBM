def test_public_imports():
    from sparklightgbm import LightGBMClassifier, LightGBMRanker, LightGBMRegressor
    assert LightGBMClassifier and LightGBMRegressor and LightGBMRanker
