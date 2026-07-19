"""
F&O Liquidity + Technicals Dashboard
-------------------------------------
100% free — no broker account, no API key, no monthly cost.
Data source: Yahoo Finance (via the yfinance library), which is free but
delayed roughly 15-20 minutes for Indian (NSE) symbols.

For real broker-grade, real-time data, you'd need a paid Data API
subscription (e.g. Dhan ~Rs 499+GST/month, Zerodha Rs 500/month) — this
free version trades a small delay for zero cost.

HOW TO RUN LOCALLY (optional, for testing on your own computer):
    pip install -r requirements.txt
    streamlit run streamlit_app.py

HOW TO DEPLOY FOR FREE (no local computer needed):
    See SETUP_GUIDE.md in this same folder for step-by-step instructions.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F&O Liquidity + Technicals",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
#  STOCK UNIVERSE  (name -> Yahoo Finance ticker, sector, tier/notes)
#  Edit this list freely to add/remove stocks.
# ─────────────────────────────────────────────────────────────
STOCKS = [
    {"name": "Reliance Industries",     "ticker": "RELIANCE.NS",   "sector": "Energy",                "tier": 1},
    {"name": "HDFC Bank",               "ticker": "HDFCBANK.NS",   "sector": "Banking & Financials",   "tier": 1},
    {"name": "ICICI Bank",              "ticker": "ICICIBANK.NS",  "sector": "Banking & Financials",   "tier": 1},
    {"name": "State Bank of India",     "ticker": "SBIN.NS",       "sector": "Banking & Financials",   "tier": 1},
    {"name": "Axis Bank",               "ticker": "AXISBANK.NS",   "sector": "Banking & Financials",   "tier": 1},
    {"name": "Kotak Mahindra Bank",     "ticker": "KOTAKBANK.NS",  "sector": "Banking & Financials",   "tier": 2},
    {"name": "Bajaj Finance",           "ticker": "BAJFINANCE.NS", "sector": "Banking & Financials",   "tier": 1},
    {"name": "Bajaj Finserv",           "ticker": "BAJAJFINSV.NS", "sector": "Banking & Financials",   "tier": 2},
    {"name": "IndusInd Bank",           "ticker": "INDUSINDBK.NS", "sector": "Banking & Financials",   "tier": 2},
    {"name": "Bank of Baroda",          "ticker": "BANKBARODA.NS", "sector": "Banking & Financials",   "tier": 2},
    {"name": "TCS",                     "ticker": "TCS.NS",        "sector": "IT",                     "tier": 1},
    {"name": "Infosys",                 "ticker": "INFY.NS",       "sector": "IT",                     "tier": 1},
    {"name": "HCL Technologies",        "ticker": "HCLTECH.NS",    "sector": "IT",                     "tier": 2},
    {"name": "Wipro",                   "ticker": "WIPRO.NS",      "sector": "IT",                     "tier": 2},
    {"name": "Tech Mahindra",           "ticker": "TECHM.NS",      "sector": "IT",                     "tier": 2},
    {"name": "ONGC",                    "ticker": "ONGC.NS",       "sector": "Energy",                 "tier": 2},
    {"name": "BPCL",                    "ticker": "BPCL.NS",       "sector": "Energy",                 "tier": 2},
    {"name": "Tata Motors",             "ticker": "TATAMOTORS.NS", "sector": "Auto",                   "tier": 1},
    {"name": "Maruti Suzuki",           "ticker": "MARUTI.NS",     "sector": "Auto",                   "tier": 1},
    {"name": "Mahindra & Mahindra",     "ticker": "M&M.NS",        "sector": "Auto",                   "tier": 1},
    {"name": "Bajaj Auto",              "ticker": "BAJAJ-AUTO.NS", "sector": "Auto",                   "tier": 2},
    {"name": "Eicher Motors",           "ticker": "EICHERMOT.NS",  "sector": "Auto",                   "tier": 2},
    {"name": "Tata Steel",              "ticker": "TATASTEEL.NS",  "sector": "Metals",                 "tier": 1},
    {"name": "JSW Steel",               "ticker": "JSWSTEEL.NS",   "sector": "Metals",                 "tier": 2},
    {"name": "Hindalco",                "ticker": "HINDALCO.NS",   "sector": "Metals",                 "tier": 2},
    {"name": "Vedanta",                 "ticker": "VEDL.NS",       "sector": "Metals",                 "tier": 2},
    {"name": "ITC",                     "ticker": "ITC.NS",        "sector": "FMCG",                   "tier": 1},
    {"name": "Hindustan Unilever",      "ticker": "HINDUNILVR.NS", "sector": "FMCG",                   "tier": 2},
    {"name": "Nestle India",            "ticker": "NESTLEIND.NS",  "sector": "FMCG",                   "tier": 2},
    {"name": "Bharti Airtel",           "ticker": "BHARTIARTL.NS", "sector": "Telecom & Infra",        "tier": 1},
    {"name": "Larsen & Toubro",         "ticker": "LT.NS",         "sector": "Telecom & Infra",        "tier": 1},
    {"name": "Sun Pharma",              "ticker": "SUNPHARMA.NS",  "sector": "Pharma",                 "tier": 2},
    {"name": "Dr. Reddy's Labs",        "ticker": "DRREDDY.NS",    "sector": "Pharma",                 "tier": 2},
    {"name": "Adani Enterprises",       "ticker": "ADANIENT.NS",   "sector": "Diversified",            "tier": 2},
    {"name": "Adani Ports",             "ticker": "ADANIPORTS.NS", "sector": "Diversified",            "tier": 2},
    {"name": "NTPC",                    "ticker": "NTPC.NS",       "sector": "PSU & Power",            "tier": 2},
    {"name": "Power Grid",              "ticker": "POWERGRID.NS",  "sector": "PSU & Power",            "tier": 2},
]

RSI_LEN = 14
ADX_LEN = 14

# ─────────────────────────────────────────────────────────────
#  INDICATOR MATH  (Wilder's smoothing — same style as the
#  Pine Script dashboard you already use on TradingView)
# ─────────────────────────────────────────────────────────────
def rsi(close: pd.Series, length: int = RSI_LEN) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx_di(high: pd.Series, low: pd.Series, close: pd.Series, length: int = ADX_LEN):
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / length, min_periods=length, adjust=False
    ).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / length, min_periods=length, adjust=False
    ).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    return adx, plus_di, minus_di


# ─────────────────────────────────────────────────────────────
#  DATA FETCH  (cached so we don't hammer Yahoo Finance —
#  refreshes automatically every 5 minutes)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_timeframe_data(ticker: str, interval: str, period: str):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def compute_all_technicals(ticker: str):
    """Returns dict: {'5M': {...}, '1H': {...}, 'Daily': {...}}"""
    tf_map = {
        "5M": ("5m", "5d"),
        "1H": ("1h", "1mo"),
        "Daily": ("1d", "6mo"),
    }
    out = {}
    for label, (interval, period) in tf_map.items():
        df = fetch_timeframe_data(ticker, interval, period)
        if df is None or len(df) < ADX_LEN + 2:
            out[label] = None
            continue
        r = rsi(df["Close"])
        a, dip, dim = adx_di(df["High"], df["Low"], df["Close"])
        out[label] = {
            "rsi": r.iloc[-1],
            "adx": a.iloc[-1],
            "dip": dip.iloc[-1],
            "dim": dim.iloc[-1],
        }
    return out


# ─────────────────────────────────────────────────────────────
#  STYLING HELPERS
# ─────────────────────────────────────────────────────────────
def rsi_color(v):
    if v is None or pd.isna(v):
        return "#8a94a6"
    if v > 60:
        return "#2fd88a"
    if v < 40:
        return "#ff5c6a"
    return "#c9d1de"


def di_display(dip, dim):
    if dip is None or dim is None or pd.isna(dip) or pd.isna(dim):
        return "—", "#8a94a6"
    if dip >= dim:
        return f"DI+ {dip:.0f} ▲", "#2fd88a"
    return f"DI- {dim:.0f} ▼", "#ff5c6a"


# ─────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0a0e14; }
    .block-container { padding-top: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 F&O Liquidity + Technicals Dashboard")
st.caption(
    "Free, self-refreshing every 5 minutes · Data via Yahoo Finance "
    "(≈15-20 min delayed) · Not investment advice — verify on your broker "
    "terminal before trading."
)

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    search = st.text_input("🔍 Search a stock", "")
with col2:
    sectors = ["All"] + sorted({s["sector"] for s in STOCKS})
    sector_choice = st.selectbox("Sector", sectors)
with col3:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

filtered = [
    s for s in STOCKS
    if (sector_choice == "All" or s["sector"] == sector_choice)
    and search.lower() in s["name"].lower()
]

if not filtered:
    st.info("No match — try a different search or sector.")
else:
    progress = st.progress(0.0, text="Fetching live-ish data…")
    rows = []
    for i, s in enumerate(filtered):
        tech = compute_all_technicals(s["ticker"])
        rows.append({"stock": s, "tech": tech})
        progress.progress((i + 1) / len(filtered), text=f"Loaded {s['name']}")
    progress.empty()

    for row in rows:
        s, tech = row["stock"], row["tech"]
        with st.container(border=True):
            top = st.columns([3, 2, 1])
            top[0].markdown(f"**{s['name']}**  \n<span style='color:#8a94a6;font-size:0.85em'>{s['sector']}</span>", unsafe_allow_html=True)
            tier_color = "#2fd88a" if s["tier"] == 1 else "#d9a63d"
            top[1].markdown(f"<span style='color:{tier_color};font-weight:700'>TIER {s['tier']}</span>", unsafe_allow_html=True)
            top[2].markdown(f"`{s['ticker']}`")

            cards = st.columns(3)
            for col, label in zip(cards, ["5M", "1H", "Daily"]):
                with col:
                    st.markdown(f"**{label}**")
                    t = tech.get(label) if tech else None
                    if not t or pd.isna(t.get("rsi", np.nan)):
                        st.markdown("<span style='color:#8a94a6'>No data</span>", unsafe_allow_html=True)
                        continue
                    rsi_v = t["rsi"]
                    adx_v = t["adx"]
                    di_text, di_col = di_display(t["dip"], t["dim"])
                    st.markdown(
                        f"""
                        <div style='font-family:monospace;font-size:0.85em;line-height:1.7'>
                        RSI: <b style='color:{rsi_color(rsi_v)}'>{rsi_v:.2f}</b><br>
                        ADX: <b>{adx_v:.0f}</b><br>
                        DI: <b style='color:{di_col}'>{di_text}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

st.divider()
st.caption(
    "RSI: green above 60, red below 40. DI: whichever of DI+/DI- is higher is "
    "shown, green for DI+, red for DI-. ADX shown neutral (trend strength only). "
    "Data delayed ~15-20 min via Yahoo Finance — free tools trade timeliness for "
    "zero cost. For real-time, use your TradingView Pine Script dashboard or a "
    "paid broker Data API."
)
