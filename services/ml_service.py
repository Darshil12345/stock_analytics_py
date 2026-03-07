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

        # ── Waterfall with base value (E[f(x)]) and final value (f(x)) ───────
        sv1      = test_sv[0]                        # shape (n_features,)
        top_idx  = np.argsort(np.abs(sv1))[-15:][::-1]

        # Base value = mean prediction probability across training set
        try:
            base_value = float(explainer.expected_value)
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_value = float(explainer.expected_value[1])
        except Exception:
            base_value = 0.5

        final_value = base_value + float(sv1.sum())

        wf_features  = [feat_names[i] for i in top_idx]
        wf_shap_vals = sv1[top_idx]

        # Build waterfall: base → features → total
        measures = ["absolute"] + ["relative"] * len(wf_features) + ["total"]
        y_labels = [f"Base value\n({base_value:.3f})"] + wf_features + [f"f(x) = {final_value:.3f}"]
        x_values = [base_value] + list(wf_shap_vals) + [final_value]

        # Color: green for positive contributions, red for negative
        colors = ["#718096"]  # base value = grey
        for v in wf_shap_vals:
            colors.append("#48bb78" if v >= 0 else "#fc8181")
        colors.append("#667eea")  # final prediction = purple

        wf_fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=measures,
            y=y_labels,
            x=x_values,
            connector={"line": {"color": "rgba(63,63,63,0.4)", "width": 1}},
            increasing={"marker": {"color": "#48bb78"}},
            decreasing={"marker": {"color": "#fc8181"}},
            totals={"marker":    {"color": "#667eea"}},
        ))
        wf_fig.add_vline(x=0.5, line_dash="dash", line_color="gray",
                         annotation_text="Decision boundary (0.5)",
                         annotation_position="top")
        wf_fig.update_layout(
            title=f"SHAP Waterfall — Sample #0 | Base: {base_value:.3f} → Prediction: {final_value:.3f}",
            xaxis_title="Probability of ↑ Up",
            xaxis=dict(range=[0, 1]),
            height=500,
        )

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

        # Intercept and prediction probabilities
        intercept   = exp.intercept[1]
        local_pred  = exp.local_pred[0]
        actual_prob = clf.predict_proba(Xte_t[idx:idx+1])[0][1]

        colors = ["#48bb78" if w > 0 else "#fc8181" for w in lime_df["LIME Weight"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=lime_df["LIME Weight"], y=lime_df["Feature"],
            orientation="h", marker_color=colors,
            name="Feature contribution",
        ))
        fig.add_vline(x=intercept, line_dash="dot", line_color="#718096", line_width=2,
                      annotation_text=f"Base rate: {intercept:.3f}",
                      annotation_position="top left",
                      annotation_font_color="#718096")
        fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)

        pos_sum = lime_df[lime_df["LIME Weight"] > 0]["LIME Weight"].sum()
        neg_sum = lime_df[lime_df["LIME Weight"] < 0]["LIME Weight"].sum()

        fig.update_layout(
            title=(
                f"LIME Explanation — Test Sample #{idx} | "
                f"Model P(Up): {actual_prob:.3f} | "
                f"LIME pred: {local_pred:.3f} | "
                f"{'UP' if actual_prob >= 0.5 else 'DOWN'}"
            ),
            xaxis_title="LIME Weight  (green = pushes Up, red = pushes Down)",
            yaxis={"categoryorder": "total ascending"},
            height=500,
            annotations=[
                dict(x=0.02, y=1.06, xref="paper", yref="paper",
                     text=f"Green sum: +{pos_sum:.4f}  |  Red sum: {neg_sum:.4f}",
                     showarrow=False, font=dict(size=11)),
            ]
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