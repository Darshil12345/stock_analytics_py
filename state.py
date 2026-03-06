"""
state.py — Single source of truth for all mutable runtime state.

All modules import from here instead of using bare globals.
In a production system replace with Redis / a proper store.
"""
from __future__ import annotations
import copy
import pandas as pd
from config import DEFAULT_INDICES, ALL_AVAILABLE_INDICES

# ── Data ──────────────────────────────────────────────────────────────────────
MASTER_DF: pd.DataFrame | None = None
NEWS_DF:   pd.DataFrame | None = None

# ── App configuration ─────────────────────────────────────────────────────────
APP_MODE:         str       = "default"
SELECTED_INDICES: list[str] = list(DEFAULT_INDICES.keys())
SELECTED_FEATURES: list[str] = ["OHLC", "Returns", "Ratios", "Time"]
DERIVED_FEATURES:  list[str] = []

# ── Index registry (mutable — user can add custom tickers) ────────────────────
INDICES_REGISTRY: dict[str, str] = copy.copy(ALL_AVAILABLE_INDICES)

# ── Last trained artefacts (used by SHAP / LIME callbacks) ───────────────────
LAST_PIPELINE     = None          # sklearn Pipeline
LAST_X_TRAIN      = None          # pd.DataFrame (filtered)
LAST_X_TEST       = None          # pd.DataFrame (filtered)
LAST_Y_TRAIN      = None          # pd.Series
LAST_Y_TEST       = None          # pd.Series
LAST_MODEL_NAME:  str | None = None
LAST_PROB_CUTOFF: float = 0.5