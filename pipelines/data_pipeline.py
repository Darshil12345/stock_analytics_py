"""
pipelines/data_pipeline.py — Orchestrates data fetching → master frame → state update.
"""
from __future__ import annotations

import pandas as pd
import state
from config import DEFAULT_INDICES, MASTER_PATH
from services.data_service import build_master


def run(
    start_date: str,
    end_date: str,
    mode: str | None = None,
    sel_indices: list[str] | None = None,
    sel_features: list[str] | None = None,
    derived_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build master DataFrame, persist to state and CSV.
    Returns the built DataFrame.
    """
    mode            = mode            or state.APP_MODE
    sel_indices     = sel_indices     or state.SELECTED_INDICES
    sel_features    = sel_features    or state.SELECTED_FEATURES
    derived_features = derived_features or state.DERIVED_FEATURES

    indices_dict = (
        DEFAULT_INDICES
        if mode == "default"
        else {k: v for k, v in state.INDICES_REGISTRY.items() if k in sel_indices}
    )

    df = build_master(start_date, end_date, indices_dict, sel_features, derived_features)

    # Create classification target if NIFTY present
    if "NIFTY_Open" in df.columns and "NIFTY_Close" in df.columns:
        df["Nifty_Open_Dir"] = (
            df["NIFTY_Open"] > df["NIFTY_Close"].shift(1)
        ).astype(int)

    # Persist
    state.MASTER_DF = df
    df.to_csv(MASTER_PATH, index=False)

    return df
