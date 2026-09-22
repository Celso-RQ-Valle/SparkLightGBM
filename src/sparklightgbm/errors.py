class SparkLightGBMError(RuntimeError):
    """Base exception for SparkLightGBM runtime failures."""


class SparkLightGBMConfigurationError(SparkLightGBMError, ValueError):
    """Raised when a requested execution configuration is invalid."""


class SparkLightGBMNetworkError(SparkLightGBMError):
    """Raised when native LightGBM workers cannot initialize networking."""


class SparkLightGBMWorkerError(SparkLightGBMError):
    """Raised when an executor-side LightGBM worker fails."""
