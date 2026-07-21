"""
F&O Liquidity + Full Technicals Dashboard
-------------------------------------------
100% free — no broker account, no API key, no monthly cost.

ARCHITECTURE (updated): this app no longer calls Yahoo Finance directly.
A separate script (update_data.py) runs on a schedule via GitHub Actions —
on GitHub's own servers, independent of anyone having this page open — and
writes results to data/latest.json. This app just reads that file. That
means the data stays fresh even if nobody has the dashboard open for hours,
which a browser-tab-only auto-refresh could never guarantee.

IMPORTANT — one-time setup: set GITHUB_RAW_URL below to your own repo's
raw.githubusercontent.com URL for data/latest.json (see SETUP_GUIDE.md).

The technicals panel below each stock shows a trimmed version of the Pine
Script's table:
    TF | H | GMMA | WT | ADX | DI | RSI
for Day / 1H / 5M, using the same indicator logic (see indicators.py).
(STCR and SF 4-Factor were removed by request to reduce computation load.)

HOW TO RUN LOCALLY (optional, for testing on your own computer):
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import time
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timezone, timedelta
from streamlit_autorefresh import st_autorefresh

import indicators as ind
from stock_list import STOCKS

# ─────────────────────────────────────────────────────────────
#  ⚠ ONE-TIME SETUP — replace with YOUR GitHub username/repo
# ─────────────────────────────────────────────────────────────
GITHUB_RAW_URL = "https://raw.githubusercontent.com/gohelnikunj/FNO-scanning/main/data/latest.json"

IST = timezone(timedelta(hours=5, minutes=30))

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jashvi FNO Scanner",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
#  STOCK UNIVERSE — imported from stock_list.py (shared with update_data.py)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
#  DATA LOADING — reads the background updater's output instead of
#  calling Yahoo Finance directly. Cached briefly so a burst of page
#  interactions doesn't hammer GitHub, but short enough that a fresh
#  Action run shows up quickly.
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_dataset(_cache_buster: int):
    try:
        # GitHub's raw-content CDN caches responses for a few minutes at the
        # edge and doesn't reliably honor a Cache-Control request header to
        # bypass that. Appending a changing query parameter makes the CDN
        # treat it as a distinct resource, which does reliably force a fresh
        # fetch from origin.
        url = f"{GITHUB_RAW_URL}?cb={_cache_buster}"
        resp = requests.get(url, timeout=10, headers={"Cache-Control": "no-cache"})
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def parse_generated_at(iso_str):
    if not iso_str:
        return None
    try:
        return datetime.fromisoformat(iso_str)
    except Exception:
        return None


def minutes_ago(dt):
    if dt is None:
        return None
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    return (now - dt).total_seconds() / 60


# ─────────────────────────────────────────────────────────────
#  RENDER THE PINE-STYLE TABLE (HTML, matches the original look)
# ─────────────────────────────────────────────────────────────
TF_ROW_COLOR = {"Day": "#e0b050", "1H": "#5888d0", "5M": "#50b878"}
TF_BG_COLOR = {"Day": "#0a1420", "1H": "#080e16", "5M": "#060c12"}


def render_pine_table(tech: dict) -> str:
    header_bg = "#0d1b2a"
    cols = ["TF", "H", "GMMA", "WT", "ADX", "DI", "RSI"]
    header_colors = ["#ffffff", "#82b1ff", "#ce93d8", "#ce93d8", "#82b1ff", "#82b1ff", "#ffcc80"]

    # NOTE: nowrap + horizontal scroll wrapper keeps every timeframe row
    # fully side-by-side on mobile instead of wrapping/stacking — user
    # swipes sideways if the screen is narrow, rather than columns
    # collapsing under each other.
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
            html += f"<td colspan='6' style='padding:4px 7px;text-align:center;color:#c9d1de;border:1px solid #1e3048;'>Not enough data</td></tr>"
            continue

        # H
        html += f"<td style='padding:4px 7px;text-align:center;color:{ind.hc_col(t['hc'])};border:1px solid #1e3048;'>{ind.DOT}</td>"
        # GMMA
        html += f"<td style='padding:4px 7px;text-align:center;color:{ind.dir_col(t['gmma_dir'])};border:1px solid #1e3048;'>{ind.gmma_txt(t['gmma_dir'], t['gmma_bars'])}</td>"
        # WT
        html += f"<td style='padding:4px 7px;text-align:center;color:{ind.dir_col(t['wt_dir'])};border:1px solid #1e3048;'>{ind.wt_tri_txt(t['wt_dir'], t['wt_bars'], t['wt_cval'], t['wt_ob'], t['wt_os'])}</td>"
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
# (saved via query param — see "💾 Save as default" below). Note: this now
# controls how often the PAGE re-checks data/latest.json (cheap, safe) — it
# no longer controls how often Yahoo Finance is called, since that's the
# background updater's job now, on its own independent schedule.
saved_refresh = st.query_params.get("refresh", "15 Min")
if saved_refresh not in REFRESH_OPTIONS:
    saved_refresh = "15 Min"
default_index = list(REFRESH_OPTIONS.keys()).index(saved_refresh)

topA, topB = st.columns([2, 3])
with topA:
    refresh_label = st.selectbox("⏱ Check for new data every", list(REFRESH_OPTIONS.keys()), index=default_index)
    if st.button("💾 Save as default"):
        st.query_params["refresh"] = refresh_label
        st.success(f"Saved. Bookmark this page's URL now — that's what makes {refresh_label} open by default next time.")
refresh_seconds = REFRESH_OPTIONS[refresh_label]

st_autorefresh(interval=refresh_seconds * 1000, key="auto_refresh_tick")

if "manual_cache_buster" not in st.session_state:
    st.session_state["manual_cache_buster"] = 0

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
        st.session_state["manual_cache_buster"] += 1
        st.session_state["just_refreshed"] = True
        st.rerun()

if st.session_state.get("just_refreshed"):
    st.toast("✓ Re-checked GitHub for the latest data", icon="🔄")
    st.session_state["just_refreshed"] = False

cache_buster = int(time.time() // 60) * 1000 + st.session_state["manual_cache_buster"]
dataset, load_err = load_dataset(cache_buster)

# ── Data freshness banner ──
if load_err:
    st.error(
        f"⚠ Couldn't reach the data file on GitHub: {load_err}  \n"
        "Check that GITHUB_RAW_URL at the top of streamlit_app.py points to "
        "your actual repo, and that data/latest.json exists there."
    )
    st.stop()

generated_dt = parse_generated_at(dataset.get("generated_at")) if dataset else None
if generated_dt is None:
    st.warning(
        "⏳ No data yet — the background updater (GitHub Actions) hasn't completed "
        "its first run. This can take up to 15 minutes after setup, or trigger it "
        "manually: your GitHub repo → Actions tab → 'Update FNO Dashboard Data' → "
        "'Run workflow'."
    )
    st.stop()

age_min = minutes_ago(generated_dt)
freshness_color = "#2fd88a" if age_min < 20 else ("#d9a63d" if age_min < 60 else "#ff5c6a")
freshness_note = "" if age_min < 20 else "  — this looks stale; check the Actions tab on GitHub for errors."
st.markdown(
    f"<div style='background:#11151d;border:1px solid {freshness_color};border-radius:8px;"
    f"padding:8px 14px;margin-bottom:10px;font-family:\"JetBrains Mono\",monospace;font-size:12.5px;'>"
    f"<span style='color:{freshness_color};font-weight:700;'>●</span> "
    f"Background data last updated: <b style='color:{freshness_color}'>{generated_dt.strftime('%d-%b %H:%M:%S')}</b> "
    f"({age_min:.0f} min ago){freshness_note}"
    f"</div>",
    unsafe_allow_html=True,
)
st.caption(
    "Not investment advice — verify on your broker terminal before trading. "
    "Data is fetched by a background job on GitHub (see update_data.py), "
    "independent of whether this page is open."
)

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
    # Read each visible stock's precomputed technicals straight out of the
    # already-loaded dataset — no network calls happen here at all, so this
    # is instant and carries zero rate-limit risk.
    enriched = []
    all_errors = {}
    stocks_data = dataset.get("stocks", {})
    for s in filtered:
        entry = stocks_data.get(s["ticker"])
        if entry is None:
            enriched.append({"stock": s, "tech": {"Day": None, "1H": None, "5M": None}, "ok": False})
            all_errors[s["name"]] = {"—": "Not present in the latest dataset (new stock? wait for next Action run)"}
            continue
        enriched.append({"stock": s, "tech": entry.get("tech", {}), "ok": entry.get("ok", False)})
        if not entry.get("ok", True):
            all_errors[s["name"]] = entry.get("errors", {})

    if all_errors:
        with st.expander(f"⚠ Debug info — {len(all_errors)} stock(s) failed in the last background run", expanded=False):
            st.caption(
                "If you see '429', 'rate limit', or 'Too Many Requests' below, "
                "Yahoo Finance temporarily rate-limited the background job — it "
                "will retry automatically on the next scheduled run."
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
                        updated_html = "<span class='updated-badge updated-ok'>✓ OK</span>"
                    else:
                        updated_html = "<span class='updated-badge updated-fail'>⚠ Failed last run</span>"

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
    "Table columns: H (SMA breakout dot), GMMA (Guppy oscillator cross + "
    "bars since), WT (WaveTrend cross + triangle strength), ADX/DI (trend "
    "strength + dominant direction), RSI (value + candles since crossing "
    "60/40, green >60 / red <40). Data is fetched and computed by a "
    "background job on GitHub (update_data.py) on its own schedule, "
    "independent of this page — see the freshness banner above for exactly "
    "when it last ran. For real-time data, use your TradingView Pine "
    "Script indicator."
)
