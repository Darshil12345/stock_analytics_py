"""
utils/indicators.py — Pure-function financial indicators.
No Dash / FastAPI imports allowed here.
"""
import numpy as np
import pandas as pd


def compute_cumulative_returns(df: pd.DataFrame, indices: list[str]) -> pd.DataFrame:
    cum = pd.DataFrame({"Date": df["Date"]})
    for idx in indices:
        col = f"{idx}_Return"
        if col in df.columns:
            daily = df[col] / 100.0
            cum[f"{idx}_Cumulative"] = (1 + daily).cumprod() - 1
    return cum


def compute_rolling_vol(
    df: pd.DataFrame, indices: list[str], window: int = 30
) -> pd.DataFrame:
    rv = pd.DataFrame({"Date": df["Date"]})
    for idx in indices:
        col = f"{idx}_Return"
        if col in df.columns:
            daily = df[col] / 100.0
            rv[f"{idx}_Vol{window}"] = daily.rolling(window).std() * np.sqrt(252)
    return rv


def compute_drawdown(df: pd.DataFrame, indices: list[str]) -> pd.DataFrame:
    dd = pd.DataFrame({"Date": df["Date"]})
    cum = compute_cumulative_returns(df, indices)
    for idx in indices:
        col = f"{idx}_Cumulative"
        if col in cum.columns:
            running_max = cum[col].cummax()
            dd[f"{idx}_Drawdown"] = (cum[col] - running_max) / (running_max + 1e-10)
    return dd
