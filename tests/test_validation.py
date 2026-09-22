import pytest

def test_missing_columns_have_clear_error():
    from sparklightgbm import LightGBMRegressor
    class Empty:
        columns = []
    with pytest.raises(ValueError, match="Missing Spark DataFrame columns"):
        LightGBMRegressor().fit(Empty())


def test_num_workers_defaults_to_auto_and_validates_overrides():
    from sparklightgbm import LightGBMRegressor
    assert LightGBMRegressor().num_workers is None
    assert LightGBMRegressor(num_workers=3).num_workers == 3
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="num_workers"):
            LightGBMRegressor(num_workers=invalid)
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="prediction_batch_size"):
            LightGBMRegressor(prediction_batch_size=invalid)


def test_early_stopping_requires_validation_data():
    from sparklightgbm import LightGBMRegressor
    from sparklightgbm.errors import SparkLightGBMConfigurationError
    estimator = LightGBMRegressor(early_stopping_rounds=2)
    with pytest.raises(SparkLightGBMConfigurationError, match="requires validation_data"):
        from sparklightgbm._execution import train_native
        train_native(estimator, object())
