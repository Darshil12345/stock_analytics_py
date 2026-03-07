"""
callbacks/sentiment_callbacks.py — News upload / fetch → sentiment UI callbacks.
Includes tag-chip custom stopwords UI.
"""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import Input, Output, State, html, dcc, dash_table, ctx, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import state
from pipelines.news_pipeline import run_google, run_upload
from utils.preprocessing import generate_wordcloud_base64, STOPWORDS_SET


def register(app):

    # ── Toggle upload vs google ───────────────────────────────────────────────
    @app.callback(
        [Output("news-upload-area", "style"),
         Output("news-google-area", "style")],
        Input("news-source", "value"),
    )
    def toggle_news_source(src):
        if src == "google":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    # ── Stopword chip management ──────────────────────────────────────────────
    @app.callback(
        [Output("stopwords-store",  "data"),
         Output("stopwords-chips",  "children"),
         Output("stopwords-hidden", "value"),
         Output("custom-stopwords", "value")],   # clear input after add
        [Input("btn-add-stopword",  "n_clicks"),
         Input("custom-stopwords",  "n_submit"),
         Input({"type": "remove-sw", "index": dash.ALL}, "n_clicks")],
        [State("custom-stopwords",  "value"),
         State("stopwords-store",   "data")],
        prevent_initial_call=True,
    )
    def manage_stopwords(add_clicks, n_submit, remove_clicks, input_val, current_words):
        import dash
        triggered = ctx.triggered_id

        words = list(current_words or [])

        # ── Remove chip ───────────────────────────────────────────────────────
        if isinstance(triggered, dict) and triggered.get("type") == "remove-sw":
            word_to_remove = triggered["index"]
            words = [w for w in words if w != word_to_remove]

        # ── Add word ──────────────────────────────────────────────────────────
        elif triggered in ("btn-add-stopword", "custom-stopwords"):
            if input_val and input_val.strip():
                # Support comma-separated input — add all at once
                new_words = [w.strip().lower() for w in input_val.split(",") if w.strip()]
                for w in new_words:
                    if w and w not in words:
                        words.append(w)

        # ── Build chip display ────────────────────────────────────────────────
        chips = []
        for word in words:
            is_base = word in STOPWORDS_SET
            chips.append(
                dbc.Badge(
                    [
                        html.Span(word, style={"marginRight": "6px"}),
                        html.Span("×", id={"type": "remove-sw", "index": word},
                                  n_clicks=0,
                                  style={"cursor": "pointer", "fontWeight": "bold",
                                         "fontSize": "14px", "lineHeight": "1"}),
                    ],
                    color="secondary" if is_base else "primary",
                    pill=True,
                    style={"padding": "6px 10px", "fontSize": "13px",
                           "display": "inline-flex", "alignItems": "center",
                           "gap": "4px"},
                    title="Already in base stopwords" if is_base else "Custom stopword",
                )
            )

        if not chips:
            chips = [html.Span("No custom stopwords added yet.",
                               style={"color": "#a0aec0", "fontSize": "13px",
                                      "fontStyle": "italic"})]

        hidden_val = ",".join(words)
        clear_input = ""   # clear the text input after adding
        return words, chips, hidden_val, clear_input

    # ── Analyze sentiment ─────────────────────────────────────────────────────
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
         State("stopwords-hidden", "value")],   # ← reads from chip store
        prevent_initial_call=True,
    )
    def process_news(_, src, contents, filename,
                     news_start, news_end, news_term, news_max, stopwords_val):
        _empty = ("", "", go.Figure(), "", "")

        # Parse hidden comma-separated stopwords value
        custom_sw = (
            {w.strip().lower() for w in stopwords_val.split(",") if w.strip()}
            if stopwords_val else None
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

            # Preview table
            preview = dash_table.DataTable(
                data=df[["raw_text","sentiment","score"]].head(20).to_dict("records"),
                columns=[{"name": c, "id": c} for c in ["raw_text","sentiment","score"]],
                style_table={"overflowX":"auto"},
                style_cell={"whiteSpace":"normal","textAlign":"left","maxWidth":"500px"},
                style_header={"backgroundColor":"#667eea","color":"white","fontWeight":"600"},
                page_size=10,
            )

            # Distribution chart
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

            # Show which custom stopwords were applied
            sw_info = ""
            if custom_sw:
                sw_info = f" | Custom stopwords applied: {', '.join(sorted(custom_sw))}"

            status = dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"✅ {label}{sw_info}",
            ], color="success")

            return status, preview, wc_src, dist_fig, badge, full_tbl

        except Exception as exc:
            import traceback; traceback.print_exc()
            return (dbc.Alert(f"❌ {exc}", color="danger"), *_empty)