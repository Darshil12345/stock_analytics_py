"""
services/sentiment_service.py — News fetching and sentiment scoring.
"""
from __future__ import annotations
import time
import urllib.parse
from datetime import datetime, date

import pandas as pd
import requests
import feedparser
from dateutil import parser as dateparser

# ticker → readable keyword for Google News
_TICKER_MAP = {
    "^NSEI": "NIFTY 50", "^DJI": "Dow Jones", "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ", "^FTSE": "FTSE 100", "^GDAXI": "DAX",
    "^N225": "Nikkei", "^HSI": "Hang Seng", "^KS11": "KOSPI",
    "^STI": "STI Singapore", "^AORD": "ASX 200", "^VIX": "VIX volatility",
    "^BVSP": "Bovespa", "^MXX": "IPC Mexico", "^IBEX": "IBEX 35",
    "^AEX": "AEX Amsterdam", "^FCHI": "CAC 40", "GC=F": "Gold price",
    "CL=F": "Crude oil price", "BTC-USD": "Bitcoin",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _to_date(val) -> date | None:
    if val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _parse_entry_date(entry: dict) -> date | None:
    for field in ("published_parsed", "updated_parsed"):
        val = entry.get(field)
        if val:
            try:
                return datetime.fromtimestamp(time.mktime(val)).date()
            except Exception:
                pass
    for field in ("published", "updated", "pubDate"):
        val = entry.get(field, "")
        if val:
            try:
                return dateparser.parse(val).date()
            except Exception:
                pass
    return None


def fetch_google_news(
    query: str,
    start_date,
    end_date,
    max_articles: int = 200,
) -> pd.DataFrame:
    """Fetch from Google News RSS (no API key needed)."""
    sd = _to_date(start_date)
    ed = _to_date(end_date)

    query_clean = _TICKER_MAP.get(
        query.strip(),
        query.strip().replace("^", "").replace("=F", "").replace("-USD", ""),
    )
    queries = [
        query_clean,
        f"{query_clean} stock market",
        f"{query_clean} index",
        f"{query_clean} shares today",
    ]

    collected: list[dict] = []
    seen: set[str] = set()

    def _in_range(d: date | None) -> bool:
        if d is None:
            return True           # unknown date → keep
        if sd and ed:
            return sd <= d <= ed
        return True

    for q in queries:
        if len(collected) >= max_articles:
            break
        url = (
            f"https://news.google.com/rss/search?"
            f"q={urllib.parse.quote(q)}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            for entry in feedparser.parse(resp.content).get("entries", []):
                if len(collected) >= max_articles:
                    break
                title   = entry.get("title", "") or ""
                summary = entry.get("summary", "") or ""
                link    = entry.get("link", "") or ""
                raw     = f"{title} {summary}".strip()
                if not raw or raw in seen:
                    continue
                pub = _parse_entry_date(entry)
                if not _in_range(pub):
                    continue
                seen.add(raw)
                collected.append({
                    "raw_text":  raw,
                    "published": str(pub) if pub else "Unknown",
                    "link":      link,
                })
        except Exception as exc:
            print(f"[sentiment_service] RSS error for '{q}': {exc}")

    # Fallback: retry without date filter
    if not collected:
        url = (
            f"https://news.google.com/rss/search?"
            f"q={urllib.parse.quote(query_clean)}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            for entry in feedparser.parse(resp.content).get("entries", []):
                if len(collected) >= max_articles:
                    break
                raw = f"{entry.get('title','')} {entry.get('summary','')}".strip()
                pub = _parse_entry_date(entry)
                if raw and raw not in seen:
                    seen.add(raw)
                    collected.append({
                        "raw_text":  raw,
                        "published": str(pub) if pub else "Unknown",
                        "link":      entry.get("link", ""),
                    })
        except Exception as exc:
            raise RuntimeError(f"RSS fetch failed for '{query_clean}': {exc}")

    if not collected:
        return pd.DataFrame(columns=["raw_text", "published", "link"])

    return (
        pd.DataFrame(collected)
        .drop_duplicates(subset=["raw_text"])
        .reset_index(drop=True)
        .head(max_articles)
    )
