import pytest
from types import SimpleNamespace


@pytest.mark.parametrize("version", ["3.4.0", "3.5.9", "4.0.0", "4.1.3", "4.2.0", "4.2.0.dev0", "4.0.0-vendor.1"])
def test_runtime_accepts_supported_spark_and_vendor_versions(monkeypatch, version):
    from sparklightgbm._validation import require_runtime
    spark = SimpleNamespace(__version__=version)
    native = object()
    monkeypatch.setattr("sparklightgbm._validation.importlib.import_module", lambda name: spark if name == "pyspark" else native)
    assert require_runtime() == (spark, native)


@pytest.mark.parametrize("version", ["2.4.8", "3.3.4", "unknown"])
def test_runtime_rejects_old_or_unrecognized_spark(monkeypatch, version):
    from sparklightgbm._validation import require_runtime
    monkeypatch.setattr("sparklightgbm._validation.importlib.import_module", lambda name: SimpleNamespace(__version__=version))
    with pytest.raises(RuntimeError, match="requires PySpark >=3.4"):
        require_runtime()

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
