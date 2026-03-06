"""
config.py — Shared constants and configuration.
All mutable runtime state lives in state.py.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
for _d in (DATA_DIR, f"{DATA_DIR}/raw", f"{DATA_DIR}/processed", "logs"):
    os.makedirs(_d, exist_ok=True)

MASTER_PATH = os.path.join(DATA_DIR, "master.csv")

# ── Index universe ─────────────────────────────────────────────────────────────
ALL_AVAILABLE_INDICES: dict[str, str] = {
    # Americas
    "NIFTY": "^NSEI", "DJI": "^DJI", "IXIC": "^IXIC", "SPX": "^GSPC",
    "RUT": "^RUT", "BOVESPA": "^BVSP", "IPC_MEXICO": "^MXX",
    # Europe
    "FTSE": "^FTSE", "GDAXI": "^GDAXI", "FCHI": "^FCHI",
    "IBEX": "^IBEX", "AEX": "^AEX",
    # Asia-Pacific
    "NIKKEI": "^N225", "HANGSENG": "^HSI", "KOSPI": "^KS11",
    "STI": "^STI", "AORD": "^AORD",
    # Volatility & Commodities
    "VIX": "^VIX", "GOLD": "GC=F", "CRUDE": "CL=F", "BITCOIN": "BTC-USD",
}

DEFAULT_INDICES: dict[str, str] = {
    "NIFTY": "^NSEI", "NIKKEI": "^N225", "DJI": "^DJI",
    "HANGSENG": "^HSI", "IXIC": "^IXIC", "GDAXI": "^GDAXI", "VIX": "^VIX",
}

# ── External links ─────────────────────────────────────────────────────────────
EXTERNAL_LINKS: dict[str, str] = {
    "TradingView":      "https://www.tradingview.com/",
    "Yahoo Finance":    "https://finance.yahoo.com/",
    "Bloomberg":        "https://www.bloomberg.com/markets",
    "Google Finance":   "https://www.google.com/finance/",
    "NSE India":        "https://www.nseindia.com/",
    "Economic Times":   "https://economictimes.indiatimes.com/markets",
    "Investing.com":    "https://www.investing.com/",
    "MarketWatch":      "https://www.marketwatch.com/",
    "CNBC":             "https://www.cnbc.com/markets/",
    "Reuters Markets":  "https://www.reuters.com/markets/",
}

# ── ML ────────────────────────────────────────────────────────────────────────
VIF_THRESHOLD = 10
PROB_CUTOFF_DEFAULT = 0.601
SHAP_BACKGROUND_SAMPLES = 50   # KernelExplainer background size
LIME_NUM_FEATURES = 15
LIME_SAMPLE_IDX = 0            # which test row to explain
