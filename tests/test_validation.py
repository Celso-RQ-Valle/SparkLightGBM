import pytest
def test_missing_columns_have_clear_error():
    from sparklightgbm import LightGBMRegressor
    class Empty:
        columns = []
    with pytest.raises(ValueError, match="Missing Spark DataFrame columns"):
        LightGBMRegressor().fit(Empty())
