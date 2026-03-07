"""
utils/preprocessing.py — NLP preprocessing helpers for the sentiment pipeline.
"""
import re
import base64
from io import BytesIO

from wordcloud import WordCloud

# ── NLTK English stopwords (full 179-word list) ───────────────────────────────
NLTK_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","you're","you've",
    "you'll","you'd","your","yours","yourself","yourselves","he","him","his",
    "himself","she","she's","her","hers","herself","it","it's","its","itself",
    "they","them","their","theirs","themselves","what","which","who","whom",
    "this","that","that'll","these","those","am","is","are","was","were","be",
    "been","being","have","has","had","having","do","does","did","doing","a",
    "an","the","and","but","if","or","because","as","until","while","of","at",
    "by","for","with","about","against","between","into","through","during",
    "before","after","above","below","to","from","up","down","in","out","on",
    "off","over","under","again","further","then","once","here","there","when",
    "where","why","how","all","both","each","few","more","most","other","some",
    "such","no","nor","not","only","own","same","so","than","too","very","s",
    "t","can","will","just","don","don't","should","should've","now","d","ll",
    "m","o","re","ve","y","ain","aren","aren't","couldn","couldn't","didn",
    "didn't","doesn","doesn't","hadn","hadn't","hasn","hasn't","haven","haven't",
    "isn","isn't","ma","mightn","mightn't","mustn","mustn't","needn","needn't",
    "shan","shan't","shouldn","shouldn't","wasn","wasn't","weren","weren't",
    "won","won't","wouldn","wouldn't",
}

# ── Finance-specific stopwords (domain noise) ─────────────────────────────────
FINANCE_STOPWORDS = {
    "said","say","says","index","nifty","market","india","session","trade",
    "trading","close","open","crore","rs","percent","stock","shares","points",
    "today","day","week","month","year","friday","monday","tuesday","wednesday",
    "thursday","saturday","sunday","quarter","annual","report","reuters",
    "bloomberg","economic","times","hindu","mint","news","press","release",
}

# Combined base set used by default
COMMON_STOPWORDS = NLTK_STOPWORDS  # alias for wordcloud (lighter)
STOPWORDS_SET    = NLTK_STOPWORDS | FINANCE_STOPWORDS

# ── Sentiment signal words (must survive stopword filtering) ──────────────────
POSITIVE_WORDS = {
    "good","great","gain","rise","bullish","positive","strong","high","record",
    "rally","surge","boost","climb","soar","optimistic","recovery","outperform",
    "buy","bull","profit","growth","upbeat","beat","exceed","robust","rebound",
}

NEGATIVE_WORDS = {
    "bad","fall","loss","bearish","negative","weak","low","crash","decline",
    "plunge","drop","tumble","slump","correct","downgrade","pessimistic",
    "underperform","sell","bear","risk","concern","disappoint","miss","below",
}

_SUFFIXES = ["ing", "ed", "ly", "es", "s"]


def _stem(token: str) -> str:
    """Minimal suffix stripping."""
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) > 2:
            return token[: -len(suf)]
    return token


def preprocess_text(
    text: str,
    stopwords_set: set = STOPWORDS_SET,
    extra_stopwords: set | None = None,
) -> str:
    """
    Clean → tokenise → stem → filter stopwords.
    Stopwords checked AFTER stemming so plurals/inflections are caught.
    extra_stopwords (user-provided) are ALWAYS removed regardless of
    sentiment-word overlap.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    result = []
    for t in text.split():
        if len(t) <= 2:
            continue
        stemmed = _stem(t)
        # User custom stopwords — always removed (both raw and stemmed form)
        if extra_stopwords and (t in extra_stopwords or stemmed in extra_stopwords):
            continue
        # Base stopword set checked on both raw and stemmed token
        if t in stopwords_set or stemmed in stopwords_set:
            continue
        result.append(stemmed)

    return " ".join(result)


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
    wc = WordCloud(
        width=800, height=400, background_color="white", max_words=200
    ).generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")