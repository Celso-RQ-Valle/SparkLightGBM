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
