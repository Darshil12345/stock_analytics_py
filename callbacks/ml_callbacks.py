"""
callbacks/ml_callbacks.py — ML training, predictions table, forecast, SHAP, LIME, comparison.
VIF removed. Predictions table and Forecast section added.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from sklearn.metrics import accuracy_score, roc_curve, auc

import state
from models.model_registry import MODEL_MAP
from pipelines.ml_pipeline import run as run_ml_pipeline, run_forecast
from services.ml_service import compute_shap, compute_lime, build_pipeline
from services.feature_service import create_training_features, filter_vif
from utils.helpers import (
    plot_decision_tree_to_base64, logistic_coef_plot, naive_bayes_priors_plot,
    gb_importance_plot, adaboost_weights_plot, svc_pca_plot, knn_k_accuracy_plot,
)


def register(app):

    # ── Train model ───────────────────────────────────────────────────────────
    @app.callback(
        [Output("ml-status",             "children"),
         Output("ml-metrics",            "children"),
         Output("roc-curve",             "figure"),
         Output("confusion-matrix",      "figure"),
         Output("model-specific-output", "children"),
         Output("predictions-table",     "children"),
         Output("cr-table-container",    "children"),
         Output("shap-section",          "children"),
         Output("lime-section",          "children")],
        Input("btn-train", "n_clicks"),
        [State("train-start",  "date"),
         State("train-end",    "date"),
         State("test-start",   "date"),
         State("test-end",     "date"),
         State("ml-model",     "value"),
         State("prob-cutoff",  "value"),
         State("chk-run-shap", "value"),
         State("chk-run-lime", "value")],
        prevent_initial_call=True,
    )
    def train_model_cb(_, train_start, train_end, test_start, test_end,
                       model_name, cutoff, run_shap, run_lime):
        empty = [go.Figure()] * 2 + [html.Div()] * 6
        if state.MASTER_DF is None:
            return (dbc.Alert("No data — build master first", color="danger"), *empty)

        try:
            results = run_ml_pipeline(
                model_name=model_name,
                train_start=train_start, train_end=train_end,
                test_start=test_start,   test_end=test_end,
                prob_cutoff=float(cutoff),
                run_shap=bool(run_shap),
                run_lime=bool(run_lime),
            )

            acc      = results["accuracy"]
            roc_auc  = results["roc_auc"]
            fpr      = results["fpr"]
            tpr      = results["tpr"]
            cm       = results["confusion_matrix"]
            cr_df    = results["classification_report"]
            pipe     = results["pipeline"]
            pred_df  = results["predictions_df"]
            X_tr     = state.LAST_X_TRAIN
            y_tr     = state.LAST_Y_TRAIN

            # ── Metrics card ──────────────────────────────────────────────
            metrics = dbc.Card(dbc.CardBody([
                html.H5("📊 Performance Metrics"), html.Hr(),
                dbc.Row([
                    dbc.Col([html.H4(model_name),        html.P("Model",    className="text-muted")], style={"textAlign":"center"}, width=4),
                    dbc.Col([html.H4(f"{acc:.2%}"),       html.P("Accuracy", className="text-muted")], style={"textAlign":"center"}, width=4),
                    dbc.Col([html.H4(f"{roc_auc:.3f}"),   html.P("ROC AUC",  className="text-muted")], style={"textAlign":"center"}, width=4),
                ]),
            ]))

            # ── ROC ───────────────────────────────────────────────────────
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={roc_auc:.3f}",
                                          line=dict(color="#667eea", width=3)))
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                          line=dict(dash="dash", color="gray"), name="Random"))
            roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")

            # ── Confusion matrix ──────────────────────────────────────────
            cm_fig = go.Figure(go.Heatmap(
                z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"],
                text=cm, texttemplate="%{text}", colorscale="Purples",
            ))
            cm_fig.update_layout(title="Confusion Matrix")

            # ── Model-specific viz ────────────────────────────────────────
            model_viz = _model_specific(model_name, pipe, X_tr, y_tr)

            # ── Predictions table ─────────────────────────────────────────
            pred_table = dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-table me-2"),
                    f"📋 Test Set Predictions ({len(pred_df)} rows)",
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Badge(f"✅ Correct: {pred_df['Result'].str.contains('Correct').sum()}",  color="success", className="me-2 p-2")),
                        dbc.Col(dbc.Badge(f"❌ Wrong:   {pred_df['Result'].str.contains('Wrong').sum()}",    color="danger",  className="me-2 p-2")),
                        dbc.Col(dbc.Badge(f"🎯 Accuracy: {acc:.2%}", color="primary", className="p-2")),
                    ], className="mb-3"),
                    dash_table.DataTable(
                        data=pred_df.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in pred_df.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "center", "padding": "8px", "fontSize": "13px"},
                        style_header={"backgroundColor": "#667eea", "color": "white", "fontWeight": "600"},
                        style_data_conditional=[
                            {"if": {"filter_query": '{Result} contains "Correct"'},
                             "backgroundColor": "#f0fff4", "color": "#276749"},
                            {"if": {"filter_query": '{Result} contains "Wrong"'},
                             "backgroundColor": "#fff5f5", "color": "#c53030"},
                            {"if": {"filter_query": '{Predicted Label} = "↑ Up"'},
                             "fontWeight": "bold"},
                        ],
                        page_size=20,
                        filter_action="native",
                        sort_action="native",
                        export_format="csv",
                    ),
                ]),
            ], className="mt-3")

            # ── Classification report ─────────────────────────────────────
            cr_table = dbc.Card(dbc.CardBody([
                html.H5("📋 Classification Report"),
                dash_table.DataTable(
                    data=cr_df.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in cr_df.columns],
                    style_header={"backgroundColor": "#667eea", "color": "white"},
                ),
            ]))

            # ── SHAP section ──────────────────────────────────────────────
            shap_section = html.Div()
            if run_shap and "shap_bar" in results:
                shap_section = html.Div([
                    html.H4("🔬 SHAP Explainability", className="section-header"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=results["shap_bar"]),       width=6),
                        dbc.Col(dcc.Graph(figure=results["shap_waterfall"]), width=6),
                    ]),
                ])

            # ── LIME section ──────────────────────────────────────────────
            lime_section = html.Div()
            if run_lime and "lime_fig" in results:
                lime_section = html.Div([
                    html.H4("🍋 LIME Local Explanation", className="section-header"),
                    dcc.Graph(figure=results["lime_fig"]),
                ])

            status = dbc.Alert(
                f"✅ Trained {model_name} | Acc: {acc:.2%} | AUC: {roc_auc:.3f} | "
                f"Test rows: {len(pred_df)}",
                color="success",
            )
            return (status, metrics, roc_fig, cm_fig,
                    model_viz, pred_table, cr_table,
                    shap_section, lime_section)

        except Exception as exc:
            import traceback; traceback.print_exc()
            return (dbc.Alert(f"❌ {exc}", color="danger"),
                    "", go.Figure(), go.Figure(),
                    html.Div(), html.Div(), html.Div(), html.Div(), html.Div())

    # ── Forecast callback ─────────────────────────────────────────────────────
    @app.callback(
        [Output("forecast-status", "children"),
         Output("forecast-table",  "children"),
         Output("forecast-chart",  "figure")],
        Input("btn-forecast", "n_clicks"),
        [State("forecast-start", "date"),
         State("forecast-end",   "date")],
        prevent_initial_call=True,
    )
    def forecast_cb(_, fcast_start, fcast_end):
        if state.LAST_PIPELINE is None:
            return dbc.Alert("Train a model first", color="warning"), html.Div(), go.Figure()
        try:
            df = run_forecast(fcast_start, fcast_end)

            if df.empty:
                return dbc.Alert("No forecast data returned — try a different date range.",
                                  color="warning"), html.Div(), go.Figure()

            # Summary badges
            up_count   = (df["Predicted Label"] == "↑ Up").sum()
            down_count = (df["Predicted Label"] == "↓ Down").sum()
            high_conf  = (df["Confidence"] == "High").sum()

            summary = dbc.Row([
                dbc.Col(dbc.Badge(f"↑ Up days: {up_count}",        color="success", className="p-2 me-2")),
                dbc.Col(dbc.Badge(f"↓ Down days: {down_count}",    color="danger",  className="p-2 me-2")),
                dbc.Col(dbc.Badge(f"High confidence: {high_conf}", color="primary", className="p-2")),
            ], className="mb-3")

            # Table
            tbl = dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df.columns],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", "padding": "8px", "fontSize": "13px"},
                style_header={"backgroundColor": "#667eea", "color": "white", "fontWeight": "600"},
                style_data_conditional=[
                    {"if": {"filter_query": '{Predicted Label} = "↑ Up"'},
                     "backgroundColor": "#f0fff4", "color": "#276749", "fontWeight": "bold"},
                    {"if": {"filter_query": '{Predicted Label} = "↓ Down"'},
                     "backgroundColor": "#fff5f5", "color": "#c53030", "fontWeight": "bold"},
                    {"if": {"filter_query": '{Confidence} = "High"'},
                     "fontWeight": "bold"},
                ],
                page_size=20,
                sort_action="native",
                export_format="csv",
            )

            # Chart
            color_map = {"↑ Up": "#48bb78", "↓ Down": "#fc8181"}
            chart = px.bar(
                df, x="Date", y="Probability",
                color="Predicted Label",
                color_discrete_map=color_map,
                title=f"Forecast: {fcast_start} → {fcast_end}",
                labels={"Probability": "P(↑ Up)"},
            )
            chart.add_hline(y=0.5, line_dash="dash", line_color="gray",
                             annotation_text="50% threshold")
            chart.update_layout(template="plotly_white")

            status = dbc.Alert(
                f"✅ Forecast for {len(df)} trading days | "
                f"↑ {up_count} up / ↓ {down_count} down | "
                f"High confidence: {high_conf} days",
                color="success",
            )
            return status, html.Div([summary, tbl]), chart

        except Exception as exc:
            import traceback; traceback.print_exc()
            return dbc.Alert(f"❌ {exc}", color="danger"), html.Div(), go.Figure()

    # ── On-demand SHAP ────────────────────────────────────────────────────────
    @app.callback(
        Output("shap-ondemand-output", "children"),
        Input("btn-shap", "n_clicks"),
        prevent_initial_call=True,
    )
    def shap_ondemand(_):
        if state.LAST_PIPELINE is None:
            return dbc.Alert("Train a model first", color="warning")
        try:
            _, bar, wf = compute_shap(
                state.LAST_PIPELINE, state.LAST_X_TRAIN,
                state.LAST_X_TEST, state.LAST_MODEL_NAME,
            )
            return html.Div([dbc.Row([
                dbc.Col(dcc.Graph(figure=bar), width=6),
                dbc.Col(dcc.Graph(figure=wf),  width=6),
            ])])
        except Exception as exc:
            return dbc.Alert(f"❌ SHAP error: {exc}", color="danger")

    # ── On-demand LIME ────────────────────────────────────────────────────────
    @app.callback(
        Output("lime-ondemand-output", "children"),
        Input("btn-lime", "n_clicks"),
        State("lime-sample-idx", "value"),
        prevent_initial_call=True,
    )
    def lime_ondemand(_, sample_idx):
        if state.LAST_PIPELINE is None:
            return dbc.Alert("Train a model first", color="warning")
        try:
            _, fig = compute_lime(
                state.LAST_PIPELINE, state.LAST_X_TRAIN,
                state.LAST_X_TEST, sample_idx=int(sample_idx or 0),
            )
            return dcc.Graph(figure=fig)
        except Exception as exc:
            return dbc.Alert(f"❌ LIME error: {exc}", color="danger")

    # ── Model comparison ──────────────────────────────────────────────────────
    @app.callback(
        [Output("model-compare-status", "children"),
         Output("model-compare-figure", "figure"),
         Output("model-compare-table",  "children")],
        Input("btn-compare-models", "n_clicks"),
        [State("train-start",    "date"),
         State("train-end",      "date"),
         State("test-start",     "date"),
         State("test-end",       "date"),
         State("compare-models", "value")],
        prevent_initial_call=True,
    )
    def compare_models(_, train_start, train_end, test_start, test_end, models_list):
        if state.MASTER_DF is None:
            return dbc.Alert("Build master first", color="warning"), go.Figure(), ""
        try:
            df = state.MASTER_DF.copy()
            df["Date"] = pd.to_datetime(df["Date"])
            train_df = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
            test_df  = df[(df["Date"] >= test_start)  & (df["Date"] <= test_end)].copy()

            X_tr, y_tr = create_training_features(train_df)
            X_te, y_te = create_training_features(test_df)
            X_te = X_te.reindex(columns=X_tr.columns, fill_value=np.nan)

            # Drop NaN rows and align
            X_tr = X_tr.dropna(how="all")
            y_tr = y_tr.loc[X_tr.index]
            X_te = X_te.dropna(how="all")
            y_te = y_te.loc[X_te.index]

            # Silent VIF filtering
            X_tr = filter_vif(X_tr)
            X_te = X_te[X_tr.columns]

            rows = []
            for name in models_list:
                try:
                    pipe = build_pipeline(MODEL_MAP[name])
                    pipe.fit(X_tr, y_tr)
                    probs = pipe.predict_proba(X_te)[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    acc   = accuracy_score(y_te, preds)
                    try:
                        fpr, tpr, _ = roc_curve(y_te, probs)
                        roc_auc = auc(fpr, tpr)
                    except Exception:
                        roc_auc = None
                    rows.append({"Model": name, "Accuracy": acc, "ROC_AUC": roc_auc})
                except Exception:
                    rows.append({"Model": name, "Accuracy": None, "ROC_AUC": None})

            res_df = pd.DataFrame(rows)
            fig = px.bar(
                res_df.melt(id_vars="Model"),
                x="Model", y="value", color="variable",
                barmode="group", title="Model Comparison",
            )
            tbl = dash_table.DataTable(
                data=res_df.round(4).to_dict("records"),
                columns=[{"name": c, "id": c} for c in res_df.columns],
                style_header={"backgroundColor": "#667eea", "color": "white"},
            )
            return dbc.Alert("✅ Comparison done", color="success"), fig, tbl
        except Exception as exc:
            return dbc.Alert(f"❌ {exc}", color="danger"), go.Figure(), ""


# ── Model-specific viz helper ─────────────────────────────────────────────────
def _model_specific(model_name, pipe, X_tr, y_tr):
    clf  = pipe.named_steps["clf"]
    base = getattr(clf, "base_estimator_", clf)

    if model_name == "Decision Tree":
        img = plot_decision_tree_to_base64(base, X_tr.columns)
        return html.Div([html.H5("🌲 Decision Tree"),
                         html.Img(src=img, style={"max-width": "100%"})])

    if model_name == "Logistic Regression":
        return html.Div([html.H5("📊 LR Coefficients"),
                         dcc.Graph(figure=logistic_coef_plot(base, X_tr.columns))])

    if model_name == "Gaussian Naive Bayes":
        return html.Div([html.H5("📊 NB Priors"),
                         dcc.Graph(figure=naive_bayes_priors_plot(base))])

    if model_name == "Random Forest":
        if hasattr(base, "feature_importances_"):
            df = (pd.DataFrame({"Feature": X_tr.columns,
                                 "Importance": base.feature_importances_})
                  .sort_values("Importance", ascending=False).head(20))
            fig = px.bar(df, x="Importance", y="Feature", orientation="h",
                         title="RF Feature Importance", color="Importance",
                         color_continuous_scale="Purples")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            return html.Div([html.H5("📊 RF Importance"), dcc.Graph(figure=fig)])

    if model_name == "Gradient Boosting":
        return html.Div([html.H5("📊 GB Importance"),
                         dcc.Graph(figure=gb_importance_plot(base, X_tr.columns))])

    if model_name == "AdaBoost":
        return html.Div([html.H5("📊 AdaBoost Weights"),
                         dcc.Graph(figure=adaboost_weights_plot(base))])

    if model_name == "SVC (RBF)":
        return html.Div([html.H5("🔮 SVC PCA Projection"),
                         dcc.Graph(figure=svc_pca_plot(base, X_tr, y_tr))])

    if model_name.startswith("KNN"):
        try:
            from sklearn.model_selection import train_test_split
            Xt, Xv, yt, yv = train_test_split(
                X_tr.fillna(0), y_tr.loc[X_tr.index], test_size=0.2, random_state=42,
            )
            return html.Div([html.H5("📈 KNN: k vs Accuracy"),
                             dcc.Graph(figure=knn_k_accuracy_plot(Xt, yt, Xv, yv))])
        except Exception:
            pass

    return html.Div()