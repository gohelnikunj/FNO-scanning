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

import time
import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

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
#  DATA FETCH  (cached — freshness aligned to the user-chosen refresh rate
#  via a "time bucket" argument, not a fixed ttl. The bucket number only
#  changes once per chosen interval, so Streamlit's cache naturally returns
#  fresh data exactly on that cadence — 5 Min / 15 Min / 30 Min / 1 Hour.)
# ─────────────────────────────────────────────────────────────
def time_bucket(interval_seconds: int) -> int:
    now_ts = int(time.time())
    return now_ts - (now_ts % interval_seconds)


def _fetch_ohlcv_raw(ticker: str, interval: str, period: str):
    """Retries once on failure (Yahoo Finance is prone to transient rate-limit
    errors on cloud-hosted IPs) and returns the real error message so it can
    be surfaced for diagnostics instead of silently swallowed."""
    last_err = None
    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            if df is not None and not df.empty:
                return df, None
            last_err = "Yahoo returned no data (empty response)"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt == 0:
            time.sleep(0.8)
    return None, last_err


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_ohlcv(ticker: str, interval: str, period: str, _bucket: int):
    df, err = _fetch_ohlcv_raw(ticker, interval, period)
    return df, err


@st.cache_data(ttl=7200, show_spinner=False)
def compute_pine_table(ticker: str, _bucket: int):
    """
    Returns {'tech': {...}, 'fetched_at': datetime, 'ok': bool, 'errors': {tf: msg}}
    '_bucket' controls freshness: it only changes once per chosen refresh
    interval (or immediately after a manual "Refresh now" click), so this
    function only actually re-runs when data should genuinely be re-fetched.
    'ok' is False if any timeframe failed — 'errors' holds the real reason
    (e.g. a Yahoo rate-limit message) for the debug panel.
    """
    tf_specs = {
        "Day": ("1d", "1y", False),
        "1H":  ("1h", "1mo", True),
        "5M":  ("5m", "5d", True),
    }
    tech = {}
    errors = {}
    ok = True
    for label, (interval, period, intraday) in tf_specs.items():
        df, err = fetch_ohlcv(ticker, interval, period, _bucket)
        result = ind.batch(df, intraday=intraday) if df is not None else None
        tech[label] = result
        if result is None:
            ok = False
            errors[label] = err or "Unknown error"
    return {"tech": tech, "fetched_at": datetime.now(), "ok": ok, "errors": errors}


# ─────────────────────────────────────────────────────────────
#  RENDER THE PINE-STYLE TABLE (HTML, matches the original look)
# ─────────────────────────────────────────────────────────────
TF_ROW_COLOR = {"Day": "#e0b050", "1H": "#5888d0", "5M": "#50b878"}
TF_BG_COLOR = {"Day": "#0a1420", "1H": "#080e16", "5M": "#060c12"}


def render_pine_table(tech: dict) -> str:
    header_bg = "#0d1b2a"
    cols = ["TF", "H", "GMMA", "WT", "STCR", "ADX", "DI", "RSI", "SF"]
    header_colors = ["#ffffff", "#82b1ff", "#ce93d8", "#ce93d8", "#a5d6a7", "#82b1ff", "#82b1ff", "#ffcc80", "#90caf9"]

    # NOTE: nowrap + horizontal scroll wrapper keeps every timeframe row
    # fully side-by-side (all 8 columns in one line) on mobile instead of
    # wrapping/stacking — user swipes sideways if the screen is narrow,
    # rather than columns collapsing under each other.
    html = """
    <div style='overflow-x:auto;-webkit-overflow-scrolling:touch;border-radius:8px;'>
    <table style='border-collapse:collapse;font-family:"JetBrains Mono",monospace;font-size:11.5px;white-space:nowrap;width:auto;'>
    <tr>
    """
    for c, col_color in zip(cols, header_colors):
        html += f"<th style='background:{header_bg};color:{col_color};padding:4px 7px;text-align:center;border:1px solid #1e3048;'>{c}</th>"
    html += "</tr>"

    for tf_label in ["Day", "1H", "5M"]:
        t = tech.get(tf_label)
        bg = TF_BG_COLOR[tf_label]
        row_color = TF_ROW_COLOR[tf_label]
        html += f"<tr style='background:{bg};'>"
        html += f"<td style='padding:4px 7px;text-align:center;color:{row_color};font-weight:700;border:1px solid #1e3048;'>{tf_label}</td>"

        if t is None:
            html += f"<td colspan='8' style='padding:4px 7px;text-align:center;color:#c9d1de;border:1px solid #1e3048;'>Not enough data</td></tr>"
            continue

        # H
        html += f"<td style='padding:4px 7px;text-align:center;color:{ind.hc_col(t['hc'])};border:1px solid #1e3048;'>{ind.DOT}</td>"
        # GMMA
        html += f"<td style='padding:4px 7px;text-align:center;color:{ind.dir_col(t['gmma_dir'])};border:1px solid #1e3048;'>{ind.gmma_txt(t['gmma_dir'], t['gmma_bars'])}</td>"
        # WT
        html += f"<td style='padding:4px 7px;text-align:center;color:{ind.dir_col(t['wt_dir'])};border:1px solid #1e3048;'>{ind.wt_tri_txt(t['wt_dir'], t['wt_bars'], t['wt_cval'], t['wt_ob'], t['wt_os'])}</td>"
        # STCR
        html += f"<td style='padding:4px 7px;text-align:center;color:{ind.dir_col(t['stcr_dir'])};border:1px solid #1e3048;'>{ind.stcr_tri_txt(t['stcr_dir'], t['stcr_bars'], t['stcr_kv'])}</td>"
        # ADX (neutral/white — trend strength only, no direction) — colored directly on the td
        adx_txt = ind.adx_val_txt(t["adx"])
        html += f"<td style='padding:4px 7px;text-align:center;color:#f1f4f8;border:1px solid #1e3048;'>{adx_txt}</td>"
        # DI (dominant side: DI+ green + ▲, DI- red + ▼) — colored directly on the td, same pattern as RSI/GMMA/WT
        di_txt = ind.di_val_txt(t["dip"], t["dim"])
        di_c = ind.di_col(t["dip"], t["dim"])
        html += f"<td style='padding:4px 7px;text-align:center;color:{di_c};font-weight:700;border:1px solid #1e3048;'>{di_txt}</td>"
        # RSI (value + candles since crossing 60/40, matching the other columns' style)
        rsi_txt = ind.rsi_val_txt_with_bars(t["rsi"], t.get("rsi_bars_60"), t.get("rsi_bars_40"))
        html += f"<td style='padding:4px 7px;text-align:center;color:{ind.rsi_col(t['rsi'])};border:1px solid #1e3048;'>{rsi_txt}</td>"
        # SF
        html += f"<td style='padding:4px 7px;text-align:center;color:{ind.sf_col(t['sf'])};font-weight:700;border:1px solid #1e3048;'>{ind.sf_txt(t['sf'])}</td>"
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

    /* Stock name badge — transparent background, yellow border, neon blue text */
    .stApp .name-badge {
        display: inline-block;
        background-color: transparent !important;
        color: #00e5ff !important;
        font-weight: 800;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 13.5px;
        white-space: nowrap;
        border: 1px solid #d9a63d;
    }

    /* Last-updated / status badge, sized to its own text */
    .updated-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        white-space: nowrap;
        font-weight: 600;
    }
    .updated-ok   { background-color: #11221a; color: #2fd88a; border: 1px solid #1e4a34; }
    .updated-fail { background-color: #2a1414; color: #ff5c6a; border: 1px solid #4a1e1e; }

    /* Stock header row — flexbox so name badge + updated badge never stack, even on mobile */
    .stock-row {
        display: flex;
        flex-wrap: nowrap;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        width: 100%;
    }
    .stock-sector { color: #8a94a6; font-size: 11.5px; }

    @media (max-width: 480px) {
        .name-badge { font-size: 12px; padding: 3px 8px; }
        .updated-badge { font-size: 10px; padding: 3px 7px; }
        .ticker-chip { font-size: 10.5px; padding: 2px 6px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Jashvi FNO Scanner")
st.markdown("<div id='alerts_top'></div>", unsafe_allow_html=True)

REFRESH_OPTIONS = {"5 Min": 300, "15 Min": 900, "30 Min": 1800, "1 Hour": 3600}

# Default refresh rate = 15 Min, unless the person saved a different default
# (saved via query param — see "💾 Save as default" below).
saved_refresh = st.query_params.get("refresh", "15 Min")
if saved_refresh not in REFRESH_OPTIONS:
    saved_refresh = "15 Min"
default_index = list(REFRESH_OPTIONS.keys()).index(saved_refresh)

topA, topB = st.columns([2, 3])
with topA:
    refresh_label = st.selectbox("⏱ Refresh rate", list(REFRESH_OPTIONS.keys()), index=default_index)
    if st.button("💾 Save as default"):
        st.query_params["refresh"] = refresh_label
        st.success(f"Saved. Bookmark this page's URL now — that's what makes {refresh_label} open by default next time.")
refresh_seconds = REFRESH_OPTIONS[refresh_label]

st.caption(
    f"Auto-refreshing every {refresh_label} · Data via Yahoo Finance "
    "(≈15-20 min delayed) · Not investment advice — verify on your broker "
    "terminal before trading."
)

# Silently reruns the app on the chosen cadence — combined with the
# time_bucket() cache key below, this is what makes data actually
# re-fetch on that schedule, not just re-render the same numbers.
st_autorefresh(interval=refresh_seconds * 1000, key="auto_refresh_tick")
current_bucket = time_bucket(refresh_seconds)

if "manual_override_bucket" not in st.session_state:
    st.session_state["manual_override_bucket"] = 0

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
        # This is what actually forces a genuine re-fetch on demand — a plain
        # st.rerun() alone does NOT bypass the interval-based cache, which was
        # the real reason the button looked like it "did nothing" before.
        st.session_state["manual_override_bucket"] = int(time.time())
        st.rerun()

# effective_bucket = whichever is newer: the natural interval schedule, or a
# just-clicked manual override. Once real time catches up past the override,
# normal cadence silently resumes on its own.
effective_bucket = max(current_bucket, st.session_state["manual_override_bucket"])

filter_choice = st.selectbox(
    "⭐ Priority filter (matching stocks float to the top; nothing is hidden)",
    [
        "None",
        "Filter 1: RSI (Daily) > 60 AND RSI (1H) > 60 — bullish alignment",
        "Filter 2: RSI (Daily) < 40 AND RSI (1H) < 40 — bearish alignment",
    ],
)

with st.expander("🔔 Alerts", expanded=True):
    alerts_enabled = st.checkbox("Enable alerts below", value=True)
    preview_mode = st.checkbox("👁 Preview alert appearance (shows a sample, doesn't need a real match)")
    st.caption(
        "🐂 Bullish: RSI (Daily) > 60 AND RSI (1H) > 60.  🐻 Bearish: RSI (Daily) < 60 "
        "AND RSI (1H) < 60. Tap a stock name to jump straight to its table below — "
        "no scrolling needed.  \n"
        "Full Alignment Alerts (below): RSI + GMMA (D/H/5M) + ADX-DI (H/5M) all "
        "pointing the same way."
    )
    alert_placeholder = st.container()

def stock_anchor_id(name: str) -> str:
    return "stock_" + re.sub(r"[^a-zA-Z0-9]", "_", name)


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
    # Small pacing between every call (not just failures) spreads requests
    # out over time instead of bursting Yahoo Finance all at once — bursts
    # are the most common cause of the "many symbols refresh failed" pattern
    # on cloud-hosted IPs.
    progress = st.progress(0.0, text="Fetching data…")
    enriched = []
    all_errors = {}
    for i, s in enumerate(filtered):
        result = compute_pine_table(s["ticker"], effective_bucket)
        enriched.append({"stock": s, "tech": result["tech"], "fetched_at": result["fetched_at"], "ok": result["ok"]})
        if not result["ok"]:
            all_errors[s["name"]] = result["errors"]
        progress.progress((i + 1) / len(filtered), text=f"Loaded {s['name']}")
        time.sleep(0.08)
    progress.empty()

    if all_errors:
        with st.expander(f"⚠ Debug info — {len(all_errors)} stock(s) failed to refresh", expanded=False):
            st.caption(
                "If you see '429', 'rate limit', or 'Too Many Requests' below, "
                "Yahoo Finance is temporarily blocking this app's shared cloud IP "
                "— this is a known limitation of free hosting, not a bug in your "
                "settings. It usually clears on its own; if it persists for hours, "
                "consider the Angel One real-data option we discussed earlier."
            )
            for name, errs in all_errors.items():
                for tf, msg in errs.items():
                    st.text(f"{name} [{tf}]: {msg}")

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

    # ── Alert checks (Alert 1: RSI D>60 & RSI 1H>60 · Alert 2: RSI D<60 & RSI 1H<60) ──
    def alert_status(tech):
        d = tech.get("Day")
        h = tech.get("1H")
        if d is None or h is None or pd.isna(d.get("rsi", np.nan)) or pd.isna(h.get("rsi", np.nan)):
            return None, None, None
        return d["rsi"], h["rsi"], (
            1 if (d["rsi"] > 60 and h["rsi"] > 60) else
            2 if (d["rsi"] < 60 and h["rsi"] < 60) else
            0
        )

    def alignment_status(tech):
        """Alert 3/4: RSI D+H | GMMA D+H+5M | ADX-DI H+5M, all pointing the same way."""
        d, h, m = tech.get("Day"), tech.get("1H"), tech.get("5M")
        if d is None or h is None or m is None:
            return 0
        d_rsi, h_rsi = d.get("rsi", np.nan), h.get("rsi", np.nan)
        if pd.isna(d_rsi) or pd.isna(h_rsi):
            return 0
        h_dip, h_dim = h.get("dip", np.nan), h.get("dim", np.nan)
        m_dip, m_dim = m.get("dip", np.nan), m.get("dim", np.nan)
        if pd.isna(h_dip) or pd.isna(h_dim) or pd.isna(m_dip) or pd.isna(m_dim):
            return 0

        bull = (
            d_rsi > 60 and h_rsi > 60
            and d["gmma_dir"] == 1 and h["gmma_dir"] == 1 and m["gmma_dir"] == 1
            and h_dip >= h_dim and m_dip >= m_dim
        )
        bear = (
            d_rsi < 40 and h_rsi < 40
            and d["gmma_dir"] == -1 and h["gmma_dir"] == -1 and m["gmma_dir"] == -1
            and h_dip < h_dim and m_dip < m_dim
        )
        if bull:
            return 3
        if bear:
            return 4
        return 0

    def alignment_row_html(name, which):
        anchor = stock_anchor_id(name)
        if which == 3:
            bg, border, txt, label = "#11221a", "#1e4a34", "#2fd88a", "Full Bullish Alignment"
            detail = "RSI D▲ H▲ &nbsp;|&nbsp; GMMA D▲ H▲ 5▲ &nbsp;|&nbsp; ADX H▲ 5▲"
        else:
            bg, border, txt, label = "#2a1414", "#4a1e1e", "#ff5c6a", "Full Bearish Alignment"
            detail = "RSI D▼ H▼ &nbsp;|&nbsp; GMMA D▼ H▼ 5▼ &nbsp;|&nbsp; ADX H▼ 5▼"
        return (
            f"<div style='background:{bg};border:1px solid {border};border-radius:8px;"
            f"padding:8px 12px;margin-bottom:6px;font-family:JetBrains Mono,monospace;font-size:12.5px;'>"
            f"<a href='#{anchor}' style='color:{txt};font-weight:700;text-decoration:none;'>🔔 {label} — ▶ {name}</a>"
            f"<div style='color:{txt};margin-top:3px;'>{detail}</div>"
            f"</div>"
        )

    def bull_bear_link(name, color):
        anchor = stock_anchor_id(name)
        return f"<a href='#{anchor}' style='color:{color};text-decoration:none;font-family:\"JetBrains Mono\",monospace;font-size:13px;display:block;padding:4px 0;'>▶ {name}</a>"

    with alert_placeholder:
        if preview_mode:
            st.markdown("**Preview (sample data — not a real signal):**", unsafe_allow_html=True)
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown("<div style='color:#2fd88a;font-weight:800;font-size:14px;'>🐂 BULLISH (sample)</div>", unsafe_allow_html=True)
                st.markdown(bull_bear_link("Sample Bull Stock Ltd.", "#2fd88a"), unsafe_allow_html=True)
            with pcol2:
                st.markdown("<div style='color:#ff5c6a;font-weight:800;font-size:14px;'>🐻 BEARISH (sample)</div>", unsafe_allow_html=True)
                st.markdown(bull_bear_link("Sample Bear Stock Ltd.", "#ff5c6a"), unsafe_allow_html=True)
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            st.markdown(alignment_row_html("Sample Stock Ltd. (DEMO)", 3), unsafe_allow_html=True)
            st.markdown(alignment_row_html("Sample Stock Ltd. (DEMO)", 4), unsafe_allow_html=True)
        elif alerts_enabled:
            bull_hits, bear_hits, a3_hits, a4_hits = [], [], [], []
            for row in enriched:
                d_rsi, h_rsi, which = alert_status(row["tech"])
                name = row["stock"]["name"]
                if which == 1:
                    bull_hits.append(name)
                elif which == 2:
                    bear_hits.append(name)
                align = alignment_status(row["tech"])
                if align == 3:
                    a3_hits.append(name)
                elif align == 4:
                    a4_hits.append(name)

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                st.markdown(f"<div style='color:#2fd88a;font-weight:800;font-size:14px;'>🐂 BULLISH ({len(bull_hits)})</div>", unsafe_allow_html=True)
                if bull_hits:
                    st.markdown("".join(bull_bear_link(n, "#2fd88a") for n in bull_hits), unsafe_allow_html=True)
                else:
                    st.caption("No matches")
            with bcol2:
                st.markdown(f"<div style='color:#ff5c6a;font-weight:800;font-size:14px;'>🐻 BEARISH ({len(bear_hits)})</div>", unsafe_allow_html=True)
                if bear_hits:
                    st.markdown("".join(bull_bear_link(n, "#ff5c6a") for n in bear_hits), unsafe_allow_html=True)
                else:
                    st.caption("No matches")

            if a3_hits or a4_hits:
                st.markdown("<div style='margin-top:14px;font-weight:700;color:#c9d1de;'>Full Alignment Alerts</div>", unsafe_allow_html=True)
                for name in a3_hits:
                    st.markdown(alignment_row_html(name, 3), unsafe_allow_html=True)
                for name in a4_hits:
                    st.markdown(alignment_row_html(name, 4), unsafe_allow_html=True)

    # Two stocks side by side (Streamlit naturally stacks these to one
    # column on narrow mobile screens, since two full tables truly can't
    # fit legibly on a phone width).
    cols_per_row = 2
    for row_start in range(0, len(enriched), cols_per_row):
        pair = enriched[row_start: row_start + cols_per_row]
        grid = st.columns(cols_per_row)
        for col, row in zip(grid, pair):
            s, tech = row["stock"], row["tech"]
            with col:
                st.markdown(f"<div id='{stock_anchor_id(s['name'])}' style='position:relative;top:-70px;'></div>", unsafe_allow_html=True)
                with st.container(border=True):
                    match_badge = ""
                    if filter_choice != "None" and row.get("_match"):
                        match_badge = "<span style='background:#d9a63d;color:#1a1408;padding:2px 8px;border-radius:5px;font-size:11px;font-weight:700;margin-left:8px;'>MATCH</span>"

                    if row["ok"]:
                        updated_html = f"<span class='updated-badge updated-ok'>✓ Updated {row['fetched_at'].strftime('%H:%M:%S')}</span>"
                    else:
                        updated_html = "<span class='updated-badge updated-fail'>⚠ Refresh failed</span>"

                    st.markdown(
                        f"""
                        <div class='stock-row'>
                            <span class='name-badge'>{s['name']}{match_badge}</span>
                            {updated_html}
                        </div>
                        <div style='margin-top:6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:space-between;'>
                            <span>
                                <span class='stock-sector'>{s['sector']}</span>
                                &nbsp;
                                <span class='ticker-chip'>{s['ticker']}</span>
                            </span>
                            <a href='#alerts_top' style='color:#d9a63d;text-decoration:none;font-size:11px;font-family:"JetBrains Mono",monospace;white-space:nowrap;'>🔝 Return to Alert</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if "note" in s:
                        st.markdown(f"<div style='color:#8a94a6;font-size:11px;margin-top:2px;'>ⓘ {s['note']}</div>", unsafe_allow_html=True)

                    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
                    st.markdown(render_pine_table(tech), unsafe_allow_html=True)

st.divider()
st.caption(
    "Table replicates your Pine Script exactly: H (SMA breakout dot), GMMA "
    "(Guppy oscillator cross + bars since), WT (WaveTrend cross + triangle "
    "strength), STCR (Stochastic RSI cross), ADX/DI (trend strength + "
    "dominant direction), RSI (value + candles since crossing 60/40, green "
    ">60 / red <40), SF (4-factor BUY/SELL/NEU). Data delayed ~15-20 min via "
    "Yahoo Finance — free tools trade timeliness for zero cost. For "
    "real-time, use your TradingView Pine Script indicator. A stock showing "
    "'⚠ Refresh failed' means Yahoo Finance didn't return data for it on the "
    "last fetch — click '🔄 Refresh now' to retry."
)
