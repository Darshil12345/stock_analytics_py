"""
callbacks/sentiment_callbacks.py — News upload / fetch → sentiment UI callbacks.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, html, dcc, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import state
from pipelines.news_pipeline import run_google, run_upload
from utils.preprocessing import generate_wordcloud_base64


def register(app):

    @app.callback(
        [Output("news-upload-area", "style"),
         Output("news-google-area", "style")],
        Input("news-source", "value"),
    )
    def toggle_news_source(src):
        if src == "google":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    @app.callback(
        [Output("p4-status",      "children"),
         Output("p4-preview",     "children"),
         Output("wc-img",         "src"),
         Output("sentiment-dist", "figure"),
         Output("avg-score",      "children"),
         Output("p4-table",       "children")],
        Input("btn-analyze", "n_clicks"),
        [State("news-source",      "value"),
         State("upload-news",      "contents"),
         State("upload-news",      "filename"),
         State("news-start",       "date"),
         State("news-end",         "date"),
         State("news-index-term",  "value"),
         State("news-max",         "value"),
         State("custom-stopwords", "value")],
        prevent_initial_call=True,
    )
    def process_news(_, src, contents, filename,
                     news_start, news_end, news_term, news_max, custom_stop):
        _empty = ("", "", go.Figure(), "", "")

        custom_sw = (
            {w.strip().lower() for w in custom_stop.split(",") if w.strip()}
            if custom_stop else None
        )

        try:
            if src == "google":
                if not news_term or not news_term.strip():
                    return (dbc.Alert("⚠️ Enter a keyword", color="warning"), *_empty)
                df = run_google(
                    news_term.strip(), news_start, news_end,
                    int(news_max or 200), custom_sw,
                )
                label = f"Google News — {len(df)} articles for '{news_term}'"
            else:
                if contents is None:
                    return (dbc.Alert("⚠️ Upload a CSV first", color="warning"), *_empty)
                df    = run_upload(contents, custom_sw)
                label = f"{filename} — {len(df)} rows"

            if df.empty:
                return (dbc.Alert("⚠️ No articles found", color="warning"), *_empty)

            # Wordcloud
            wc_src = generate_wordcloud_base64(" ".join(df["wc_text"]))

            # Preview
            preview = dash_table.DataTable(
                data=df[["raw_text","sentiment","score"]].head(20).to_dict("records"),
                columns=[{"name": c, "id": c} for c in ["raw_text","sentiment","score"]],
                style_table={"overflowX":"auto"},
                style_cell={"whiteSpace":"normal","textAlign":"left","maxWidth":"500px"},
                style_header={"backgroundColor":"#667eea","color":"white","fontWeight":"600"},
                page_size=10,
            )

            # Distribution
            dist = df["sentiment"].value_counts().reset_index()
            dist.columns = ["Sentiment","Count"]
            color_map = {"Positive":"#48bb78","Negative":"#fc8181","Neutral":"#90cdf4"}
            dist_fig = px.bar(dist, x="Sentiment", y="Count",
                              color="Sentiment", color_discrete_map=color_map,
                              title="Sentiment Distribution", template="plotly_white")

            # Average score badge
            avg = df["score"].mean()
            badge = dbc.Badge(
                f"Average Score: {avg:.2f}",
                color="success" if avg > 0 else "danger" if avg < 0 else "secondary",
                style={"fontSize":"1.1rem","padding":"10px 20px"},
            )

            # Full table
            full_tbl = dash_table.DataTable(
                data=df[["raw_text","sentiment","score"]].head(50).to_dict("records"),
                columns=[{"name": c, "id": c} for c in ["raw_text","sentiment","score"]],
                style_table={"overflowX":"auto"},
                style_cell={"whiteSpace":"normal","textAlign":"left","maxWidth":"500px"},
                style_header={"backgroundColor":"#667eea","color":"white","fontWeight":"600"},
                filter_action="native", sort_action="native", page_size=10,
            )

            status = dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"✅ {label}",
            ], color="success")

            return status, preview, wc_src, dist_fig, badge, full_tbl

        except Exception as exc:
            import traceback; traceback.print_exc()
            return (dbc.Alert(f"❌ {exc}", color="danger"), *_empty)