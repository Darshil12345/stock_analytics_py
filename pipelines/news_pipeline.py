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
    sw = STOPWORDS_SET.copy()
    if custom_stopwords:
        sw.update(w.lower() for w in custom_stopwords if w.strip())

    # Sentiment pass: keep sentiment words to avoid muting signal
    sw_sent = sw - POSITIVE_WORDS - NEGATIVE_WORDS

    df["sent_text"] = df["raw_text"].apply(lambda x: preprocess_text(x, sw_sent))
    df["wc_text"]   = df["raw_text"].apply(lambda x: preprocess_text(x, COMMON_STOPWORDS))
    df[["sentiment", "score"]] = df["sent_text"].apply(
        lambda x: pd.Series(simple_sentiment(x))
    )

    state.NEWS_DF = df
    return df
