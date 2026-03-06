"""
services/data_service.py — Data fetching and master-frame construction.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
import yfinance as yf


# ── Single index fetch ────────────────────────────────────────────────────────
def fetch_index(name: str, ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return pd.DataFrame({
        "Date":           pd.to_datetime(df["Date"]),
        f"{name}_Open":   df["Open"],
        f"{name}_High":   df["High"],
        f"{name}_Low":    df["Low"],
        f"{name}_Close":  df["Close"],
        f"{name}_Volume": df.get("Volume", np.nan),
    })


# ── Master builder ────────────────────────────────────────────────────────────
def build_master(
    start_date: str,
    end_date: str,
    indices_dict: dict[str, str],
    feature_types: list[str],
    derived_features: list[str] | None = None,
) -> pd.DataFrame:
    frames = [
        fetch_index(name, ticker, start_date, end_date)
        for name, ticker in indices_dict.items()
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No data downloaded. Try expanding dates.")

    master = frames[0]
    for df in frames[1:]:
        master = master.merge(df, on="Date", how="outer")

    master = master.sort_values("Date").reset_index(drop=True).ffill().bfill()

    if "Returns" in feature_types:
        for name in indices_dict:
            col = f"{name}_Close"
            if col in master.columns:
                master[f"{name}_Return"] = master[col].pct_change() * 100.0

    if "Ratios" in feature_types:
        skip = {"VIX", "GOLD", "CRUDE", "BITCOIN"}
        for name in indices_dict:
            if name in skip:
                continue
            o, c = f"{name}_Open", f"{name}_Close"
            if o in master.columns and c in master.columns:
                master[f"{name}_Open_Close_Ratio"] = master[o] / master[c].shift(1)

    if "Time" in feature_types:
        master["Year"]      = master["Date"].dt.year
        master["Quarter"]   = master["Date"].dt.quarter
        master["Month"]     = master["Date"].dt.month
        master["DayOfWeek"] = master["Date"].dt.dayofweek

    for formula in (derived_features or []):
        try:
            master = _apply_derived(master, formula)
        except Exception as exc:
            print(f"[data_service] formula '{formula}' failed: {exc}")

    ret_cols = [c for c in master.columns if c.endswith("_Return")]
    if ret_cols:
        master = master.dropna(subset=ret_cols, how="all").reset_index(drop=True)

    return master


# ── Derived-feature parser ────────────────────────────────────────────────────
def _apply_derived(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    if "=" not in formula:
        return df
    feat, expr = (s.strip() for s in formula.split("=", 1))

    if "MA(" in expr:
        m = re.search(r"MA\((\w+),\s*(\d+)\)", expr)
        if m and m.group(1) in df.columns:
            df[feat] = df[m.group(1)].rolling(int(m.group(2))).mean()
            return df

    if "LAG(" in expr:
        m = re.search(r"LAG\((\w+),\s*(\d+)\)", expr)
        if m and m.group(1) in df.columns:
            df[feat] = df[m.group(1)].shift(int(m.group(2)))
            return df

    try:
        df[feat] = df.eval(expr)
    except Exception:
        pass
    return df
