"""
pipelines/news_pipeline.py — Orchestrates news fetch → preprocessing → sentiment scoring.
"""
from __future__ import annotations
import base64
from io import BytesIO

import pandas as pd

import state
from services.sentiment_service import fetch_google_news
from utils.preprocessing import (
    preprocess_text, simple_sentiment,
    STOPWORDS_SET, COMMON_STOPWORDS, POSITIVE_WORDS, NEGATIVE_WORDS,
)


def run_google(
    query: str,
    start_date,
    end_date,
    max_articles: int = 200,
    custom_stopwords: set[str] | None = None,
) -> pd.DataFrame:
    df = fetch_google_news(query, start_date, end_date, max_articles)
    return _enrich(df, custom_stopwords)


def run_upload(
    contents: str,
    custom_stopwords: set[str] | None = None,
) -> pd.DataFrame:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(BytesIO(decoded))

    candidates = ["text", "content", "article", "headline", "title", "news", "body", "description"]
    text_col = next((c for c in df.columns if c.lower() in candidates), None)
    if text_col is None:
        lengths = {c: df[c].astype(str).str.len().median() for c in df.columns}
        text_col = max(lengths, key=lengths.get)

    df["raw_text"] = df[text_col].astype(str)
    return _enrich(df, custom_stopwords)


def _enrich(df: pd.DataFrame, custom_stopwords: set[str] | None) -> pd.DataFrame:
    # Normalise custom stopwords — stem them too so e.g. "markets" → "market"
    extra_sw = None
    if custom_stopwords:
        from utils.preprocessing import _stem
        extra_sw = {_stem(w.lower()) for w in custom_stopwords if w.strip()}
        extra_sw |= {w.lower() for w in custom_stopwords if w.strip()}  # also raw form

    # For sentiment scoring: use base stopwords MINUS sentiment signal words
    # Custom stopwords are passed separately so they are ALWAYS removed
    sw_sent = STOPWORDS_SET - POSITIVE_WORDS - NEGATIVE_WORDS

    # For wordcloud: use only common stopwords (lighter filter, more visual variety)
    sw_wc = COMMON_STOPWORDS

    df["sent_text"] = df["raw_text"].apply(
        lambda x: preprocess_text(x, sw_sent, extra_stopwords=extra_sw)
    )
    df["wc_text"] = df["raw_text"].apply(
        lambda x: preprocess_text(x, sw_wc, extra_stopwords=extra_sw)
    )
    df[["sentiment", "score"]] = df["sent_text"].apply(
        lambda x: pd.Series(simple_sentiment(x))
    )

    state.NEWS_DF = df
    return df