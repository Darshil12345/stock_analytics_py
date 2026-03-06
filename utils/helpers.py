"""
utils/helpers.py — Miscellaneous helpers (tree-plot, Plotly helpers, etc.).
"""
from __future__ import annotations
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Decision-tree PNG ─────────────────────────────────────────────────────────
def plot_decision_tree_to_base64(tree_model, feature_names, max_depth: int = 3) -> str:
    from sklearn.tree import plot_tree
    try:
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            tree_model, feature_names=feature_names,
            class_names=["Dir=0", "Dir=1"], filled=True,
            rounded=True, fontsize=10, max_depth=max_depth, ax=ax,
        )
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    except Exception as exc:
        print(f"Tree plot error: {exc}")
        return ""


# ── Per-model Plotly helpers ──────────────────────────────────────────────────
def logistic_coef_plot(model, feature_names) -> go.Figure:
    try:
        coef = model.coef_[0]
        df = (
            pd.DataFrame({"Feature": feature_names, "Coefficient": coef,
                           "Abs": np.abs(coef)})
            .sort_values("Abs", ascending=False).head(20)
        )
        fig = px.bar(df, x="Coefficient", y="Feature", orientation="h",
                     title="Top 20 Logistic Regression Coefficients",
                     color="Coefficient", color_continuous_scale="RdBu_r")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        return fig
    except Exception:
        return go.Figure()


def naive_bayes_priors_plot(model) -> go.Figure:
    try:
        df = pd.DataFrame({"Class": ["Dir=0", "Dir=1"],
                            "Prior Probability": model.class_prior_})
        return px.bar(df, x="Class", y="Prior Probability",
                      title="Naive Bayes: Class Prior Probabilities",
                      color="Prior Probability", color_continuous_scale="Viridis")
    except Exception:
        return go.Figure()


def gb_importance_plot(model, feature_names) -> go.Figure:
    try:
        df = (
            pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
            .sort_values("Importance", ascending=False).head(20)
        )
        fig = px.bar(df, x="Importance", y="Feature", orientation="h",
                     title="Top 20 Feature Importances — Gradient Boosting",
                     color="Importance", color_continuous_scale="Oranges")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        return fig
    except Exception:
        return go.Figure()


def adaboost_weights_plot(model) -> go.Figure:
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=model.estimator_weights_, mode="lines+markers",
                                  name="Estimator Weights",
                                  line=dict(width=2), marker=dict(size=8)))
        fig.update_layout(title="AdaBoost: Estimator Weights",
                          xaxis_title="Estimator Index", yaxis_title="Weight")
        return fig
    except Exception:
        return go.Figure()


def svc_pca_plot(model, X: pd.DataFrame, y: pd.Series) -> go.Figure:
    try:
        Xc = X.dropna()
        if Xc.shape[0] < 3:
            return go.Figure()
        pca = PCA(n_components=2)
        XY = pca.fit_transform(Xc)
        df2 = pd.DataFrame({"PC1": XY[:, 0], "PC2": XY[:, 1],
                             "y": y.loc[Xc.index].astype(str).values,
                             "is_sv": False})
        try:
            svc = getattr(model, "base_estimator_", model)
            dec = svc.decision_function(Xc)
            k = max(5, int(0.05 * len(dec)))
            df2.iloc[np.argsort(np.abs(dec))[-k:],
                     df2.columns.get_loc("is_sv")] = True
        except Exception:
            pass
        return px.scatter(df2, x="PC1", y="PC2", color="y", symbol="is_sv",
                          title="SVC projection (PCA) — support-vector-like points marked",
                          labels={"y": "class"})
    except Exception:
        return go.Figure()


def knn_k_accuracy_plot(X_train, y_train, X_val, y_val, k_max: int = 25) -> go.Figure:
    try:
        ks = list(range(1, min(k_max, 50) + 1))
        accs = [accuracy_score(y_val, KNeighborsClassifier(n_neighbors=k)
                               .fit(X_train, y_train).predict(X_val)) for k in ks]
        return px.line(pd.DataFrame({"k": ks, "acc": accs}),
                       x="k", y="acc", title="K vs Validation Accuracy")
    except Exception:
        return go.Figure()
