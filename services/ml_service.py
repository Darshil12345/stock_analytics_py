"""
services/ml_service.py — Model training, SHAP explanations, LIME explanations, Forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report

from models.model_registry import MODEL_MAP, TREE_MODELS
from config import SHAP_BACKGROUND_SAMPLES, LIME_NUM_FEATURES, LIME_SAMPLE_IDX


# ── Pipeline builder ──────────────────────────────────────────────────────────
def build_pipeline(estimator) -> Pipeline:
    clf = (
        estimator
        if hasattr(estimator, "predict_proba")
        else CalibratedClassifierCV(estimator, cv=3)
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_dates: pd.Series,
    prob_cutoff: float = 0.5,
) -> dict:
    """
    Train a model. Returns results dict including a predictions table.
    """
    estimator = MODEL_MAP[model_name]
    pipe = build_pipeline(estimator)
    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= prob_cutoff).astype(int)

    acc = accuracy_score(y_test, preds)
    cm  = confusion_matrix(y_test, preds)

    try:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), 0.0

    cr = classification_report(y_test, preds, output_dict=True)
    cr_df = pd.DataFrame(cr).T.round(4).reset_index().rename(columns={"index": "Class"})

    # ── Predictions table ─────────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "Date":        test_dates.reset_index(drop=True).values,
        "Actual":      y_test.reset_index(drop=True).values,
        "Predicted":   preds,
        "Probability": probs.round(4),
        "Correct":     (y_test.reset_index(drop=True).values == preds),
    })
    pred_df["Actual Label"]    = pred_df["Actual"].map({1: "↑ Up", 0: "↓ Down"})
    pred_df["Predicted Label"] = pred_df["Predicted"].map({1: "↑ Up", 0: "↓ Down"})
    pred_df["Result"]          = pred_df["Correct"].map({True: "✅ Correct", False: "❌ Wrong"})
    pred_df = pred_df[["Date", "Actual Label", "Predicted Label", "Probability", "Result"]]

    return dict(
        accuracy=acc, roc_auc=roc_auc, fpr=fpr, tpr=tpr,
        confusion_matrix=cm, pipeline=pipe,
        classification_report=cr_df,
        predictions_df=pred_df,
    )


# ── Forecast (future dates outside master data) ───────────────────────────────
def forecast_future(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    forecast_start: str,
    forecast_end: str,
    prob_cutoff: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch fresh market data for forecast_start->forecast_end,
    build the same features as training, and return predictions.

    Returns DataFrame: Date | Predicted Label | Probability | Confidence
    """
    import state
    from config import DEFAULT_INDICES
    from services.data_service import fetch_index

    indices_dict = DEFAULT_INDICES if state.APP_MODE == "default" else {
        k: v for k, v in state.INDICES_REGISTRY.items() if k in state.SELECTED_INDICES
    }

    # Fetch data
    frames = []
    for name, ticker in indices_dict.items():
        dfi = fetch_index(name, ticker, forecast_start, forecast_end)
        if not dfi.empty:
            frames.append(dfi)

    if not frames:
        raise RuntimeError("No data available for the forecast date range.")

    fcast_df = frames[0]
    for df in frames[1:]:
        fcast_df = fcast_df.merge(df, on="Date", how="outer")
    fcast_df = fcast_df.sort_values("Date").reset_index(drop=True).ffill().bfill()

    # Build features
    skip = {"VIX", "GOLD", "CRUDE", "BITCOIN"}
    for name in indices_dict:
        col = f"{name}_Close"
        if col in fcast_df.columns:
            fcast_df[f"{name}_Return"] = fcast_df[col].pct_change() * 100.0
    for name in indices_dict:
        if name in skip:
            continue
        o, c = f"{name}_Open", f"{name}_Close"
        if o in fcast_df.columns and c in fcast_df.columns:
            fcast_df[f"{name}_Open_Close_Ratio"] = fcast_df[o] / fcast_df[c].shift(1)

    fcast_df["Year"]      = fcast_df["Date"].dt.year
    fcast_df["Quarter"]   = fcast_df["Date"].dt.quarter
    fcast_df["Month"]     = fcast_df["Date"].dt.month
    fcast_df["DayOfWeek"] = fcast_df["Date"].dt.dayofweek

    # Lag (same as training)
    for col in [c for c in fcast_df.columns if c.endswith("_Return") or "Ratio" in c]:
        fcast_df[col] = fcast_df[col].shift(1)

    fcast_df = fcast_df.dropna(how="all").reset_index(drop=True)
    dates    = fcast_df["Date"].copy()

    # Align to training columns
    X_fcast = fcast_df.reindex(columns=X_train.columns, fill_value=np.nan)

    if X_fcast.dropna(how="all").empty:
        raise RuntimeError("Feature matrix is empty after alignment — check date range.")

    probs = pipeline.predict_proba(X_fcast)[:, 1]
    preds = (probs >= prob_cutoff).astype(int)

    return pd.DataFrame({
        "Date":            dates.values,
        "Predicted Label": pd.Series(preds).map({1: "↑ Up", 0: "↓ Down"}).values,
        "Probability":     probs.round(4),
        "Confidence":      pd.Series(probs).apply(
            lambda p: "High"   if p >= 0.7 or p <= 0.3
                      else "Medium" if p >= 0.6 or p <= 0.4
                      else "Low"
        ).values,
    })


# ── SHAP ──────────────────────────────────────────────────────────────────────
def compute_shap(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, go.Figure, go.Figure]:
    try:
        import shap
    except ImportError:
        msg = "shap not installed — run: pip install shap"
        return pd.DataFrame(), _empty_fig(msg), _empty_fig(msg)

    imp  = pipeline.named_steps["imputer"]
    scl  = pipeline.named_steps["scaler"]
    clf  = pipeline.named_steps["clf"]
    base = getattr(clf, "base_estimator_", clf)

    Xtr_t = scl.transform(imp.transform(X_train.fillna(0)))
    Xte_t = scl.transform(imp.transform(X_test.fillna(0)))
    feat_names = list(X_train.columns)

    try:
        if model_name in TREE_MODELS:
            explainer   = shap.TreeExplainer(base)
            shap_values = explainer.shap_values(Xtr_t)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            if hasattr(shap_values, "values"):
                shap_values = shap_values.values
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

            test_sv = explainer.shap_values(Xte_t[:1])
            if isinstance(test_sv, list):
                test_sv = test_sv[1]
            if hasattr(test_sv, "values"):
                test_sv = test_sv.values
            if isinstance(test_sv, np.ndarray) and test_sv.ndim == 3:
                test_sv = test_sv[:, :, 1]
        else:
            bg = shap.sample(Xtr_t, min(SHAP_BACKGROUND_SAMPLES, len(Xtr_t)))
            explainer   = shap.KernelExplainer(base.predict_proba, bg)
            shap_values = explainer.shap_values(Xte_t[:50], nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            test_sv = shap_values[:1]

        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df  = (
            pd.DataFrame({"Feature": feat_names, "Mean |SHAP|": mean_abs})
            .sort_values("Mean |SHAP|", ascending=False)
            .head(20)
        )
        bar_fig = px.bar(
            shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
            title="SHAP — Top 20 Features by Mean |SHAP Value|",
            color="Mean |SHAP|", color_continuous_scale="Blues",
        )
        bar_fig.update_layout(yaxis={"categoryorder": "total ascending"})

        sv1     = test_sv[0]
        top_idx = np.argsort(np.abs(sv1))[-15:][::-1]
        wf_df   = pd.DataFrame({
            "Feature":    [feat_names[i] for i in top_idx],
            "SHAP Value": sv1[top_idx],
        })
        wf_fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"] * len(wf_df),
            y=wf_df["Feature"],
            x=wf_df["SHAP Value"],
            connector={"line": {"color": "rgb(63,63,63)"}},
        ))
        wf_fig.update_layout(title="SHAP Waterfall — Sample #0", xaxis_title="SHAP Value")

        return shap_df, bar_fig, wf_fig

    except Exception as exc:
        err = f"SHAP computation failed: {exc}"
        print(err)
        return pd.DataFrame(), _empty_fig(err), _empty_fig(err)


# ── LIME ──────────────────────────────────────────────────────────────────────
def compute_lime(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    sample_idx: int = LIME_SAMPLE_IDX,
    num_features: int = LIME_NUM_FEATURES,
) -> tuple[pd.DataFrame, go.Figure]:
    try:
        from lime import lime_tabular
    except ImportError:
        msg = "lime not installed — run: pip install lime"
        return pd.DataFrame(), _empty_fig(msg)

    try:
        imp = pipeline.named_steps["imputer"]
        scl = pipeline.named_steps["scaler"]
        clf = pipeline.named_steps["clf"]

        Xtr_t = scl.transform(imp.transform(X_train.fillna(0)))
        Xte_t = scl.transform(imp.transform(X_test.fillna(0)))
        feat_names = list(X_train.columns)

        explainer = lime_tabular.LimeTabularExplainer(
            Xtr_t, feature_names=feat_names,
            class_names=["Dir=0", "Dir=1"],
            mode="classification", random_state=42,
        )
        idx = min(sample_idx, len(Xte_t) - 1)
        exp = explainer.explain_instance(Xte_t[idx], clf.predict_proba, num_features=num_features)

        lime_df = (
            pd.DataFrame(exp.as_list(), columns=["Feature", "LIME Weight"])
            .assign(abs_w=lambda d: d["LIME Weight"].abs())
            .sort_values("abs_w", ascending=False)
            .drop(columns="abs_w")
        )
        colors = ["#48bb78" if w > 0 else "#fc8181" for w in lime_df["LIME Weight"]]
        fig = go.Figure(go.Bar(
            x=lime_df["LIME Weight"], y=lime_df["Feature"],
            orientation="h", marker_color=colors,
        ))
        fig.update_layout(
            title=f"LIME Explanation — Test Sample #{idx}",
            xaxis_title="LIME Weight",
            yaxis={"categoryorder": "total ascending"},
        )
        return lime_df, fig

    except Exception as exc:
        err = f"LIME computation failed: {exc}"
        print(err)
        return pd.DataFrame(), _empty_fig(err)


# ── helpers ───────────────────────────────────────────────────────────────────
def _empty_fig(msg: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                        xref="paper", yref="paper", font=dict(size=14))
    return fig