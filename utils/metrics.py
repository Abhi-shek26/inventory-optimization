"""Metric aggregation.

The implementation lives in `utils/simulation.py` so that the simulation rules
and the scoring rules sit side by side; this module keeps the historical import
path used by the experiment scripts.
"""

from utils.simulation import METRIC_COLUMNS, summarize  # noqa: F401

__all__ = ["summarize", "METRIC_COLUMNS"]
