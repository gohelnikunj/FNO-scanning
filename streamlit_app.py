"""
F&O Liquidity + Full Technicals Dashboard
-------------------------------------------
100% free — no broker account, no API key, no monthly cost.
Data source: Yahoo Finance (via the yfinance library), which is free but
delayed roughly 15-20 minutes for Indian (NSE) symbols.

The expandable "Technicals" panel below each stock reproduces your Pine
Script's exact 9-column table:
    TF | H | GMMA | WT | STCR | ADX | DI | RSI | SF
for Day / 1H / 5M, using the same indicator logic (see indicators.py).

HOW TO RUN LOCALLY (optional, for testing on your own computer):
    pip install -r requirements.txt
    streamlit run streamlit_app.py

HOW TO DEPLOY FOR FREE (no local computer needed):
    See SETUP_GUIDE.md in this same folder for step-by-step instructions.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import indicators as ind

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jashvi FNO Scanner",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
#  STOCK UNIVERSE  (name -> Yahoo Finance ticker, sector, tier)
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
    {"name": "Crude Oil (MCX proxy)",   "ticker": "CL=F",          "sector": "Commodities",             "tier": 2,
     "note": "NYMEX WTI (USD) — global proxy, not the exact MCX INR contract"},
    {"name": "Natural Gas (MCX proxy)", "ticker": "NG=F",          "sector": "Commodities",             "tier": 2,
     "note": "NYMEX Henry Hub (USD) — global proxy, not the exact MCX INR contract"},
]

# ─────────────────────────────────────────────────────────────
#  DATA FETCH  (cached — refreshes automatically every 5 minutes)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(ticker: str, interval: str, period: str):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def compute_pine_table(ticker: str):
    """Returns dict: {'Day': {...}, '1H': {...}, '5M': {...}} using indicators.batch()"""
    tf_specs = {
        "Day": ("1d", "1y", False),
        "1H":  ("1h", "1mo", True),
        "5M":  ("5m", "5d", True),
    }
    out = {}
    for label, (interval, period, intraday) in tf_specs.items():
        df = fetch_ohlcv(ticker, interval, period)
        out[label] = ind.batch(df, intraday=intraday) if df is not None else None
    return out


# ─────────────────────────────────────────────────────────────
#  RENDER THE PINE-STYLE TABLE (HTML, matches the original look)
# ─────────────────────────────────────────────────────────────
TF_ROW_COLOR = {"Day": "#e0b050", "1H": "#5888d0", "5M": "#50b878"}
TF_BG_COLOR = {"Day": "#0a1420", "1H": "#080e16", "5M": "#060c12"}


def render_pine_table(tech: dict) -> str:
    header_bg = "#0d1b2a"
    cols = ["TF", "H", "GMMA", "WT", "STCR", "ADX/DI", "RSI", "SF"]
    header_colors = ["#ffffff", "#82b1ff", "#ce93d8", "#ce93d8", "#a5d6a7", "#82b1ff", "#ffcc80", "#90caf9"]

    # NOTE: nowrap + horizontal scroll wrapper keeps every timeframe row
    # fully side-by-side (all 8 columns in one line) on mobile instead of
    # wrapping/stacking — user swipes sideways if the screen is narrow,
    # rather than columns collapsing under each other.
    html = """
    <div style='overflow-x:auto;-webkit-overflow-scrolling:touch;border-radius:8px;'>
    <table style='width:100%;min-width:520px;border-collapse:collapse;font-family:"JetBrains Mono",monospace;font-size:12.5px;white-space:nowrap;'>
    <tr>
    """
    for c, col_color in zip(cols, header_colors):
        html += f"<th style='background:{header_bg};color:{col_color};padding:6px 8px;text-align:center;border:1px solid #1e3048;'>{c}</th>"
    html += "</tr>"

    for tf_label in ["Day", "1H", "5M"]:
        t = tech.get(tf_label)
        bg = TF_BG_COLOR[tf_label]
        row_color = TF_ROW_COLOR[tf_label]
        html += f"<tr style='background:{bg};'>"
        html += f"<td style='padding:6px 8px;text-align:center;color:{row_color};font-weight:700;border:1px solid #1e3048;'>{tf_label}</td>"

        if t is None:
            html += f"<td colspan='7' style='padding:6px 8px;text-align:center;color:#c9d1de;border:1px solid #1e3048;'>Not enough data</td></tr>"
            continue

        # H
        html += f"<td style='padding:6px 8px;text-align:center;color:{ind.hc_col(t['hc'])};border:1px solid #1e3048;'>{ind.DOT}</td>"
        # GMMA
        html += f"<td style='padding:6px 8px;text-align:center;color:{ind.dir_col(t['gmma_dir'])};border:1px solid #1e3048;'>{ind.gmma_txt(t['gmma_dir'], t['gmma_bars'])}</td>"
        # WT
        html += f"<td style='padding:6px 8px;text-align:center;color:{ind.dir_col(t['wt_dir'])};border:1px solid #1e3048;'>{ind.wt_tri_txt(t['wt_dir'], t['wt_bars'], t['wt_cval'], t['wt_ob'], t['wt_os'])}</td>"
        # STCR
        html += f"<td style='padding:6px 8px;text-align:center;color:{ind.dir_col(t['stcr_dir'])};border:1px solid #1e3048;'>{ind.stcr_tri_txt(t['stcr_dir'], t['stcr_bars'], t['stcr_kv'])}</td>"
        # ADX/DI combined cell (two values, two colors)
        adx_txt = ind.adx_val_txt(t["adx"])
        di_txt = ind.di_val_txt(t["dip"], t["dim"])
        di_c = ind.di_col(t["dip"], t["dim"])
        html += f"<td style='padding:6px 8px;text-align:center;border:1px solid #1e3048;'><span style='color:#f1f4f8'>{adx_txt}</span> / <span style='color:{di_c}'>{di_txt}</span></td>"
        # RSI (value + candles since crossing 60/40, matching the other columns' style)
        rsi_txt = ind.rsi_val_txt_with_bars(t["rsi"], t.get("rsi_bars_60"), t.get("rsi_bars_40"))
        html += f"<td style='padding:6px 8px;text-align:center;color:{ind.rsi_col(t['rsi'])};border:1px solid #1e3048;'>{rsi_txt}</td>"
        # SF
        html += f"<td style='padding:6px 8px;text-align:center;color:{ind.sf_col(t['sf'])};font-weight:700;border:1px solid #1e3048;'>{ind.sf_txt(t['sf'])}</td>"
        html += "</tr>"

    html += "</table></div>"
    return html


# ─────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0a0e14; color: #e7ecf3; }
    .block-container { padding-top: 1.6rem; }

    /* Force readable text everywhere, regardless of Streamlit's light/dark theme guess */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
    .stMarkdown, .stCaption, .stMarkdown p {
        color: #e7ecf3 !important;
    }
    h1, h2, h3 { color: #ffffff !important; }

    /* Search box + selectbox + inputs */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #11151d !important;
        color: #e7ecf3 !important;
        border: 1px solid #232b38 !important;
    }
    .stTextInput label, .stSelectbox label { color: #c9d1de !important; }

    /* Buttons */
    .stButton button {
        background-color: #11151d !important;
        color: #e7ecf3 !important;
        border: 1px solid #d9a63d !important;
    }
    .stButton button:hover { border-color: #ffcc66 !important; color: #ffffff !important; }

    /* Containers / cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #11151d !important;
        border: 1px solid #232b38 !important;
    }

    /* Expander */
    .stExpander {
        background-color: #11151d !important;
        border: 1px solid #232b38 !important;
    }
    .stExpander summary, .stExpander summary p {
        color: #d9a63d !important;
        font-weight: 600 !important;
    }

    /* Info / caption boxes */
    .stAlert, .stAlert p { color: #e7ecf3 !important; }
    small, .stCaption p { color: #8a94a6 !important; }

    /* Ticker chip */
    .ticker-chip {
        background-color: #1a2130;
        color: #90caf9;
        padding: 3px 8px;
        border-radius: 5px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        white-space: nowrap;
    }

    /* Stock header row — flexbox so it never stacks, even on mobile */
    .stock-row {
        display: flex;
        flex-wrap: nowrap;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        width: 100%;
    }
    .stock-row .name-block { flex: 1 1 auto; min-width: 0; }
    .stock-row .name-block .stock-name {
        font-weight: 700; color: #ffffff; font-size: 14.5px;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display:block;
    }
    .stock-row .name-block .stock-sector { color: #8a94a6; font-size: 11.5px; }
    .stock-row .tier-badge { flex: 0 0 auto; font-weight: 700; font-size: 12px; white-space: nowrap; }

    @media (max-width: 480px) {
        .stock-row .name-block .stock-name { font-size: 13px; }
        .ticker-chip { font-size: 10.5px; padding: 2px 6px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Jashvi FNO Scanner")
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

filter_choice = st.selectbox(
    "⭐ Priority filter (matching stocks float to the top; nothing is hidden)",
    [
        "None",
        "Filter 1: RSI (Daily) > 60 AND RSI (1H) > 60 — bullish alignment",
        "Filter 2: RSI (Daily) < 40 AND RSI (1H) < 40 — bearish alignment",
    ],
)

filtered = [
    s for s in STOCKS
    if (sector_choice == "All" or s["sector"] == sector_choice)
    and search.lower() in s["name"].lower()
]

if not filtered:
    st.info("No match — try a different search or sector.")
else:
    # Precompute technicals for every visible stock up front — needed so we
    # can sort by the filter before rendering, not just while scrolling.
    progress = st.progress(0.0, text="Fetching data…")
    enriched = []
    for i, s in enumerate(filtered):
        tech = compute_pine_table(s["ticker"])
        enriched.append({"stock": s, "tech": tech})
        progress.progress((i + 1) / len(filtered), text=f"Loaded {s['name']}")
    progress.empty()

    def matches_filter(tech):
        d = tech.get("Day")
        h = tech.get("1H")
        if d is None or h is None or pd.isna(d.get("rsi", np.nan)) or pd.isna(h.get("rsi", np.nan)):
            return False, 0.0
        if filter_choice.startswith("Filter 1"):
            ok = d["rsi"] > 60 and h["rsi"] > 60
            return ok, d["rsi"] + h["rsi"]
        if filter_choice.startswith("Filter 2"):
            ok = d["rsi"] < 40 and h["rsi"] < 40
            return ok, -(d["rsi"] + h["rsi"])
        return False, 0.0

    if filter_choice != "None":
        for row in enriched:
            ok, score = matches_filter(row["tech"])
            row["_match"] = ok
            row["_score"] = score
        enriched.sort(key=lambda r: (not r["_match"], -r["_score"]))

    for row in enriched:
        s, tech = row["stock"], row["tech"]
        with st.container(border=True):
            tier_color = "#2fd88a" if s["tier"] == 1 else "#d9a63d"
            match_badge = ""
            if filter_choice != "None" and row.get("_match"):
                match_badge = "<span style='background:#d9a63d;color:#1a1408;padding:2px 8px;border-radius:5px;font-size:11px;font-weight:700;margin-left:8px;'>MATCH</span>"

            st.markdown(
                f"""
                <div class='stock-row'>
                    <div class='name-block'>
                        <span class='stock-name'>{s['name']}{match_badge}</span>
                        <span class='stock-sector'>{s['sector']}</span>
                    </div>
                    <span class='tier-badge' style='color:{tier_color}'>TIER {s['tier']}</span>
                    <span class='ticker-chip'>{s['ticker']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "note" in s:
                st.markdown(f"<div style='color:#8a94a6;font-size:11px;margin-top:2px;'>ⓘ {s['note']}</div>", unsafe_allow_html=True)

            with st.expander("📋 Full technicals (H / GMMA / WT / STCR / ADX / DI / RSI / SF)"):
                st.markdown(render_pine_table(tech), unsafe_allow_html=True)

st.divider()
st.caption(
    "Table replicates your Pine Script exactly: H (SMA breakout dot), GMMA "
    "(Guppy oscillator cross + bars since), WT (WaveTrend cross + triangle "
    "strength), STCR (Stochastic RSI cross), ADX/DI (trend strength + "
    "dominant direction), RSI (value + candles since crossing 60/40, green "
    ">60 / red <40), SF (4-factor BUY/SELL/NEU). Data delayed ~15-20 min via "
    "Yahoo Finance — free tools trade timeliness for zero cost. For "
    "real-time, use your TradingView Pine Script indicator."
)
