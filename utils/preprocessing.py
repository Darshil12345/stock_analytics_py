"""
utils/preprocessing.py — NLP preprocessing helpers for the sentiment pipeline.
"""
import re
import base64
from io import BytesIO

from wordcloud import WordCloud

# ── Stop-word sets ────────────────────────────────────────────────────────────
COMMON_STOPWORDS = {
    "the","a","an","and","or","but","is","are","was","were","be","being","been",
    "have","has","had","having","do","does","did","doing","to","of","in","for",
    "on","at","by","with","from","as","this","that","these","those","it","its",
    "i","me","my","myself","we","our","ours","you","your","yours","he","him",
    "his","she","her","hers","they","them","their","what","which","who","whom",
    "when","where","how","why","can","could","will","would","shall","should",
    "may","might","must",
}

CUSTOM_STOPWORDS = {
    "said","say","says","may","might","could","will","would","index","nifty",
    "market","high","india","support","gain","gains","rose","fall","fell","up",
    "down","flat","today","day","week","month","year","friday","session","trade",
    "trading","close","open","crore","rs","percent","stock","shares","points",
}

STOPWORDS_SET = COMMON_STOPWORDS | CUSTOM_STOPWORDS

POSITIVE_WORDS = {
    "good","great","up","gain","rise","bullish","positive","strong","high","record",
    "rally","surge","boost","climb","soar","optimistic","recovery","outperform","buy","bull",
}

NEGATIVE_WORDS = {
    "bad","fall","down","loss","bearish","negative","weak","low","crash","decline",
    "plunge","drop","tumble","slump","correct","downgrade","pessimistic","underperform","sell","bear",
}

_SUFFIXES = ["ing", "ed", "ly", "es", "s"]


def preprocess_text(text: str, stopwords_set: set = STOPWORDS_SET) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in stopwords_set and len(t) > 2]
    stemmed = []
    for t in tokens:
        for suf in _SUFFIXES:
            if t.endswith(suf) and len(t) - len(suf) > 2:
                t = t[: -len(suf)]
                break
        stemmed.append(t)
    return " ".join(stemmed)


def simple_sentiment(text: str) -> tuple[str, int]:
    if not text:
        return "Neutral", 0
    words = text.split()
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    score = pos - neg
    if score > 0:
        return "Positive", score
    elif score < 0:
        return "Negative", score
    return "Neutral", 0


def generate_wordcloud_base64(text: str) -> str:
    if not text.strip():
        return ""
    wc = WordCloud(width=800, height=400, background_color="white", max_words=200).generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
