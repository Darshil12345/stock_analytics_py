"""
api/routes_news.py — REST endpoints for news fetching and sentiment.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

import state
from pipelines.news_pipeline import run_google

router = APIRouter()


class NewsRequest(BaseModel):
    query:        str           = "NIFTY"
    start_date:   str           = "2024-01-01"
    end_date:     str           = "2024-12-31"
    max_articles: int           = 100
    custom_stopwords: Optional[str] = None   # comma-separated


@router.post("/fetch")
def fetch_news(req: NewsRequest):
    """Fetch + analyse news from Google News RSS."""
    try:
        sw = (
            {w.strip().lower() for w in req.custom_stopwords.split(",") if w.strip()}
            if req.custom_stopwords else None
        )
        df = run_google(req.query, req.start_date, req.end_date, req.max_articles, sw)
        return {
            "articles": len(df),
            "sentiment_counts": df["sentiment"].value_counts().to_dict(),
            "avg_score": round(float(df["score"].mean()), 3),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/summary")
def news_summary():
    """Return summary of the last sentiment analysis run."""
    if state.NEWS_DF is None:
        raise HTTPException(status_code=404, detail="No news analysed yet")
    df = state.NEWS_DF
    return {
        "articles": len(df),
        "sentiment_counts": df["sentiment"].value_counts().to_dict(),
        "avg_score": round(float(df["score"].mean()), 3),
    }
