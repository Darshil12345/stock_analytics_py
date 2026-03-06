"""
callbacks/eda_callbacks.py — Register Univariate / Bivariate / Multivariate callbacks.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash import Input, Output, State, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import dash_table
from sklearn.decomposition import PCA

import state
from utils.indicators import compute_cumulative_returns, compute_rolling_vol, compute_drawdown


def register(app):
    # ── Univariate ────────────────────────────────────────────────────────────
    @app.callback(
        [Output("uni-status", "children"),
         Output("uni-figure", "figure"),
         Output("uni-table", "children")],
        Input("btn-uni", "n_clicks"),
        [State("uni-index", "value"), State("uni-plot", "value")],
        prevent_initial_call=True,
    )
    def univariate_cb(_, uni_idx, uni_plot):
        if state.MASTER_DF is None:
            return dbc.Alert("Build master data first", color="warning"), go.Figure(), ""
        df = state.MASTER_DF.copy()
        try:
            if uni_plot == "timeseries":
                col = f"{uni_idx}_Close"
                if col not in df.columns:
                    return dbc.Alert(f"{col} not available", "danger"), go.Figure(), ""
                fig = px.line(df, x="Date", y=col, title=f"{uni_idx} Close Price")
                return dbc.Alert("✅ Ready", "success"), fig, ""

            if uni_plot == "hist_kde":
                col = f"{uni_idx}_Return"
                if col not in df.columns:
                    return dbc.Alert(f"{col} not available", "danger"), go.Figure(), ""
                fig = px.histogram(df.dropna(subset=[col]), x=col, marginal="box",
                                   nbins=60, title=f"{uni_idx} Return Distribution")
                return dbc.Alert("✅ Ready", "success"), fig, ""

            if uni_plot in ("box", "violin"):
                col = f"{uni_idx}_Return"
                if col not in df.columns:
                    return dbc.Alert(f"{col} not available", "danger"), go.Figure(), ""
                fn = px.box if uni_plot == "box" else px.violin
                fig = fn(df.dropna(subset=[col]), x="Year", y=col,
                         title=f"{uni_idx} Returns by Year")
                return dbc.Alert("✅ Ready", "success"), fig, ""

            if uni_plot == "rolling_vol":
                rv  = compute_rolling_vol(df, [uni_idx])
                col = f"{uni_idx}_Vol30"
                if col not in rv.columns:
                    return dbc.Alert("Cannot compute rolling vol", "danger"), go.Figure(), ""
                fig = px.line(rv, x="Date", y=col, title=f"{uni_idx} Rolling Volatility (30d)")
                return dbc.Alert("✅ Ready", "success"), fig, ""

            if uni_plot == "cumulative":
                cum = compute_cumulative_returns(df, [uni_idx])
                col = f"{uni_idx}_Cumulative"
                if col not in cum.columns:
                    return dbc.Alert("Cannot compute cumulative", "danger"), go.Figure(), ""
                fig = px.line(cum, x="Date", y=col, title=f"{uni_idx} Cumulative Returns")
                return dbc.Alert("✅ Ready", "success"), fig, ""

            if uni_plot == "drawdown":
                dd  = compute_drawdown(df, [uni_idx])
                col = f"{uni_idx}_Drawdown"
                if col not in dd.columns:
                    return dbc.Alert("Cannot compute drawdown", "danger"), go.Figure(), ""
                fig = px.line(dd, x="Date", y=col, title=f"{uni_idx} Drawdown")
                return dbc.Alert("✅ Ready", "success"), fig, ""

            if uni_plot == "stats":
                col = f"{uni_idx}_Return"
                if col not in df.columns:
                    return dbc.Alert(f"{col} not available", "danger"), go.Figure(), ""
                stats = df[col].describe().reset_index()
                stats.columns = ["Statistic", "Value"]
                tbl = dash_table.DataTable(
                    data=stats.round(4).to_dict("records"),
                    columns=[{"name": c, "id": c} for c in stats.columns],
                )
                return dbc.Alert("✅ Stats ready", "success"), go.Figure(), tbl

            return dbc.Alert("⚠️ Plot not implemented", "warning"), go.Figure(), ""
        except Exception as exc:
            return dbc.Alert(f"❌ {exc}", "danger"), go.Figure(), ""

    # ── Bivariate ─────────────────────────────────────────────────────────────
    @app.callback(
        [Output("bi-status", "children"), Output("bi-figure", "figure")],
        Input("btn-bi", "n_clicks"),
        [State("bi-index-1", "value"), State("bi-index-2", "value"),
         State("bi-plot", "value")],
        prevent_initial_call=True,
    )
    def bivariate_cb(_, i1, i2, plot):
        if state.MASTER_DF is None:
            return dbc.Alert("Build master data first", "warning"), go.Figure()
        df = state.MASTER_DF.copy()
        xcol, ycol = f"{i1}_Return", f"{i2}_Return"
        try:
            if plot == "scatter":
                fig = px.scatter(df.dropna(subset=[xcol, ycol]), x=xcol, y=ycol,
                                 title=f"{i1} vs {i2} Returns")
            elif plot == "scatter_trend":
                fig = px.scatter(df.dropna(subset=[xcol, ycol]), x=xcol, y=ycol,
                                 trendline="ols", title=f"{i1} vs {i2} Returns (trend)")
            elif plot == "correlation":
                corr = df[[xcol, ycol]].dropna().corr().iloc[0, 1]
                fig  = go.Figure(go.Indicator(
                    mode="number+delta", value=corr,
                    title={"text": f"Pearson Corr ({i1}, {i2})"},
                ))
            elif plot == "lagged_corr":
                lags = list(range(-10, 11))
                vals = [df[xcol].shift(l).corr(df[ycol]) for l in lags]
                fig  = px.line(pd.DataFrame({"lag": lags, "corr": vals}),
                               x="lag", y="corr", title=f"Lagged corr: {i1} vs {i2}")
            else:
                return dbc.Alert("⚠️ Not implemented", "warning"), go.Figure()
            return dbc.Alert("✅ Ready", "success"), fig
        except Exception as exc:
            return dbc.Alert(f"❌ {exc}", "danger"), go.Figure()

    # ── Multivariate ─────────────────────────────────────────────────────────
    @app.callback(
        [Output("multi-status", "children"), Output("multi-figure", "figure")],
        Input("btn-multi", "n_clicks"),
        [State("multi-indices", "value"), State("multi-plot", "value")],
        prevent_initial_call=True,
    )
    def multivariate_cb(_, indices, plot):
        if state.MASTER_DF is None:
            return dbc.Alert("Build master data first", "warning"), go.Figure()
        df   = state.MASTER_DF.copy()
        cols = [f"{i}_Return" for i in indices if f"{i}_Return" in df.columns]
        data = df[cols].dropna()
        try:
            if plot == "corr_matrix":
                fig = px.imshow(data.corr(), text_auto=".2f", zmin=-1, zmax=1,
                                title="Correlation Matrix")
            elif plot == "scatter_matrix":
                fig = px.scatter_matrix(data.sample(min(len(data), 500)))
                fig.update_traces(diagonal_visible=False)
            elif plot == "pca":
                if len(data) < 3:
                    return dbc.Alert("Not enough rows for PCA", "warning"), go.Figure()
                pca = PCA(n_components=2)
                c   = pca.fit_transform(data)
                fig = px.scatter(x=c[:, 0], y=c[:, 1], labels={"x": "PC1", "y": "PC2"},
                                 title=f"PCA (var: {pca.explained_variance_ratio_.sum():.2%})")
            elif plot == "multi_compare":
                fig = go.Figure()
                for idx in indices:
                    col = f"{idx}_Close"
                    if col in df.columns:
                        fig.add_trace(go.Scatter(x=df["Date"], y=df[col],
                                                  name=idx, mode="lines"))
                fig.update_layout(title="Multi-index Close Price Comparison")
            else:
                return dbc.Alert("⚠️ Not implemented", "warning"), go.Figure()
            return dbc.Alert("✅ Ready", "success"), fig
        except Exception as exc:
            return dbc.Alert(f"❌ {exc}", "danger"), go.Figure()
