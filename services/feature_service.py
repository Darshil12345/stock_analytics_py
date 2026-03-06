"""
services/feature_service.py — ML feature engineering and VIF filtering.
VIF is used internally to remove multicollinear features but never displayed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ── Feature matrix builder ────────────────────────────────────────────────────
def create_training_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    df = df.copy()

    # Lag returns and ratios by 1 day (prevent look-ahead)
    for col in [c for c in df.columns if c.endswith("_Return") or "Ratio" in c]:
        df[col] = df[col].shift(1)

    # Target: NIFTY opens higher than previous close
    if "Nifty_Open_Dir" not in df.columns:
        if "NIFTY_Open" in df.columns and "NIFTY_Close" in df.columns:
            df["Nifty_Open_Dir"] = (
                df["NIFTY_Open"] > df["NIFTY_Close"].shift(1)
            ).astype(int)

    # Time features
    if "Year" not in df.columns:
        df["Year"]      = df["Date"].dt.year
        df["Quarter"]   = df["Date"].dt.quarter
        df["Month"]     = df["Date"].dt.month
        df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Exclude raw OHLCV columns (keep engineered ones)
    raw_ohlcv = {
        c for c in df.columns
        if any(x in c for x in ("_Open", "_High", "_Low", "_Close", "_Volume"))
        and "Ratio" not in c
    }
    exclude = {"Date", "Nifty_Open_Dir"} | raw_ohlcv
    X_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[X_cols].copy()
    y = df["Nifty_Open_Dir"].copy() if "Nifty_Open_Dir" in df.columns else None
    return X, y


# ── VIF (internal use only — filters multicollinear features, not displayed) ──
def _compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.select_dtypes(include=[np.number]).dropna(axis=1, how="all").dropna()
    if Xc.shape[1] < 2:
        return pd.DataFrame(columns=["Feature", "VIF"])
    rows = [
        {"Feature": Xc.columns[i], "VIF": variance_inflation_factor(Xc.values, i)}
        for i in range(Xc.shape[1])
    ]
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def filter_vif(X: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """Remove multicollinear features. Returns filtered X only."""
    vif_df = _compute_vif(X)
    if vif_df.empty:
        return X

    good = vif_df.loc[vif_df["VIF"] < threshold, "Feature"].tolist()
    if not good:
        good = vif_df.nsmallest(10, "VIF")["Feature"].tolist()

    return X[good].copy()