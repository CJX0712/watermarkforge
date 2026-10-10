"""WatermarkForge error hierarchy (E100..E500).

Author: 晨星
"""

from __future__ import annotations


class WmError(Exception):
    """Base error for WatermarkForge."""


class ConfigError(WmError):
    """E100: invalid configuration / env override."""


class DataError(WmError):
    """E200: invalid synthetic data generation request."""


class SchemeError(WmError):
    """E300: watermark scheme mis-use (e.g. unknown scheme)."""


class DetectError(WmError):
    """E400: detector mis-use (e.g. unknown detector)."""


class EvalError(WmError):
    """E500: evaluation / benchmark mis-use."""
