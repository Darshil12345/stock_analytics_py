"""
pipelines/ml_pipeline.py — Feature engineering → VIF → training → explainability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import state
from services.feature_service import create_training_features, filter_vif
from services.ml_service import train_model, compute_shap, compute_lime, forecast_future


def _prepare(df: pd.DataFrame):
    """
    Build X, y from a date-sliced master df.
    Always returns positional (0-based) index so X and y are always aligned.
    """
    X, y = create_training_features(df)

    # Reset to positional index — eliminates ALL alignment bugs
    X = X.reset_index(drop=True)
    if y is not None:
        y = y.reset_index(drop=True)

    # Drop rows where y is NaN (first row after shift loses target)
    if y is not None:
        valid = y.notna()
        X = X.loc[valid].reset_index(drop=True)
        y = y.loc[valid].reset_index(drop=True)

    # Also drop rows where every feature is NaN
    not_all_nan = ~X.isna().all(axis=1)
    X = X.loc[not_all_nan].reset_index(drop=True)
    if y is not None:
        y = y.loc[not_all_nan].reset_index(drop=True)

    return X, y


def run(
    model_name: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    prob_cutoff: float = 0.5,
    run_shap: bool = False,
    run_lime: bool = False,
) -> dict:
    if state.MASTER_DF is None:
        raise RuntimeError("Master data not built yet.")

    df = state.MASTER_DF.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    train_df = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
    test_df  = df[(df["Date"] >= test_start)  & (df["Date"] <= test_end)].copy()

    if train_df.empty:
        raise RuntimeError("No training data in the selected date range.")
    if test_df.empty:
        raise RuntimeError("No test data in the selected date range.")

    X_train, y_train = _prepare(train_df)
    X_test,  y_test  = _prepare(test_df)

    if y_train is None or y_train.empty:
        raise RuntimeError("Target column (Nifty_Open_Dir) missing — check NIFTY is in your selected indices.")
    if y_test is None or y_test.empty:
        raise RuntimeError("Test target is empty — try a wider test date range.")

    # Align test columns to train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan)

    # VIF filtering — silent, train columns only
    X_train = filter_vif(X_train)
    X_test  = X_test[X_train.columns]

    # Pull test dates (positional — matches y_test after _prepare)
    test_dates = test_df["Date"].reset_index(drop=True).iloc[: len(y_test)]

    results = train_model(
        model_name, X_train, y_train,
        X_test, y_test, test_dates, prob_cutoff,
    )

    pipe = results["pipeline"]
    state.LAST_PIPELINE    = pipe
    state.LAST_X_TRAIN     = X_train
    state.LAST_X_TEST      = X_test
    state.LAST_Y_TRAIN     = y_train
    state.LAST_Y_TEST      = y_test
    state.LAST_MODEL_NAME  = model_name
    state.LAST_PROB_CUTOFF = prob_cutoff

    if run_shap:
        shap_df, shap_bar, shap_wf = compute_shap(pipe, X_train, X_test, model_name)
        results.update(shap_df=shap_df, shap_bar=shap_bar, shap_waterfall=shap_wf)

    if run_lime:
        lime_df, lime_fig = compute_lime(pipe, X_train, X_test)
        results.update(lime_df=lime_df, lime_fig=lime_fig)

    return results


def run_forecast(forecast_start: str, forecast_end: str) -> pd.DataFrame:
    if state.LAST_PIPELINE is None:
        raise RuntimeError("Train a model first before forecasting.")
    return forecast_future(
        pipeline=state.LAST_PIPELINE,
        X_train=state.LAST_X_TRAIN,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        prob_cutoff=getattr(state, "LAST_PROB_CUTOFF", 0.5),
    )