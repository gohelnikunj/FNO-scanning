"""
indicators.py
--------------
A faithful Python port of the Pine Script's indicator functions, so the
Streamlit dashboard can reproduce a matching table:
(TF | H | GMMA | WT | ADX | DI | RSI) per stock, per timeframe.
(STCR and SF 4-Factor were removed by request to simplify computation.)

A couple of things are approximated because they don't translate 1:1 outside
TradingView's bar-replay engine — flagged with "APPROX" comments below.
"""

import numpy as np
import pandas as pd

DOT = "●"


# ─────────────────────────────────────────────────────────────
#  BASIC MOVING AVERAGES
# ─────────────────────────────────────────────────────────────
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing — matches Pine's ta.rma."""
    return series.ewm(alpha=1 / length, adjust=False).mean()


def rsi_series(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    return rma(tr, length)


# ─────────────────────────────────────────────────────────────
#  GENERIC CROSS-STATE TRACKER
#  Mirrors the Pine pattern: once a→b cross happens, direction persists
#  and a bar counter increments until the opposite cross occurs.
# ─────────────────────────────────────────────────────────────
def cross_state(a: pd.Series, b: pd.Series):
    a = a.values
    b = b.values
    n = len(a)
    direction = 0
    bars = 0
    for i in range(1, n):
        if np.isnan(a[i]) or np.isnan(b[i]) or np.isnan(a[i - 1]) or np.isnan(b[i - 1]):
            continue
        bull = a[i - 1] <= b[i - 1] and a[i] > b[i]
        bear = a[i - 1] >= b[i - 1] and a[i] < b[i]
        if bull:
            direction, bars = 1, 0
        elif bear:
            direction, bars = -1, 0
        elif direction != 0:
            bars += 1
    return direction, bars


# ─────────────────────────────────────────────────────────────
#  H-CONDITION
# ─────────────────────────────────────────────────────────────
def h_condition(df: pd.DataFrame, sma_len: int = 10) -> int:
    sma_val = sma(df["Close"], sma_len)
    buy = (df["Close"] > sma_val) & (df["Close"] > df["High"].shift(1))
    sell = (df["Close"] < sma_val) & (df["Close"] < df["Low"].shift(1))
    if bool(buy.iloc[-1]):
        return 1
    if bool(sell.iloc[-1]):
        return -1
    return 0


# ─────────────────────────────────────────────────────────────
#  GMMA (GUPPY) OSCILLATOR
# ─────────────────────────────────────────────────────────────
def gmma_state(close: pd.Series, smooth: int = 1, signal: int = 13):
    fast = ema(close, 3) + ema(close, 5) + ema(close, 8) + ema(close, 10) + ema(close, 12) + ema(close, 15)
    slow = ema(close, 30) + ema(close, 35) + ema(close, 40) + ema(close, 45) + ema(close, 50) + ema(close, 60)
    osc_raw = (fast - slow) / slow.replace(0, np.nan) * 100
    osc = osc_raw.rolling(smooth).mean() if smooth > 1 else osc_raw
    sig = ema(osc_raw, signal)
    return cross_state(osc, sig)


# ─────────────────────────────────────────────────────────────
#  ADX + DI+ / DI-
# ─────────────────────────────────────────────────────────────
def adx_di(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atrv = atr_series(high, low, close, length)
    dip = 100 * rma(pd.Series(plus_dm, index=high.index), length) / atrv.replace(0, np.nan)
    dim = 100 * rma(pd.Series(minus_dm, index=high.index), length) / atrv.replace(0, np.nan)
    dx = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    adxv = rma(dx, length)
    return adxv, dip, dim


# ─────────────────────────────────────────────────────────────
#  WAVETREND
# ─────────────────────────────────────────────────────────────
def wavetrend(df: pd.DataFrame, ch: int, avg: int, ma_len: int, ob: float, os_: float):
    src = (df["High"] + df["Low"] + df["Close"]) / 3
    esa = ema(src, ch)
    de = ema((src - esa).abs(), ch)
    ci = (src - esa) / (0.015 * de.replace(0, np.nan))
    wt1 = ema(ci, avg)
    wt2 = sma(wt1, ma_len)
    direction, bars = cross_state(wt1, wt2)
    cval = wt1.iloc[-1]
    return direction, bars, cval, ob, os_


# ─────────────────────────────────────────────────────────────
#  RSI THRESHOLD-CROSS BAR COUNTER
#  "n" = candles since RSI last crossed the given level in the given
#  direction — same style as the GMMA/WT "bars since cross" cells.
#  Returns None if that crossing never happened in the fetched window
#  (caller should just show the plain RSI value in that case).
# ─────────────────────────────────────────────────────────────
def rsi_cross_bars(rsi: pd.Series, level: float, direction: str):
    values = rsi.values
    n = len(values)
    last_cross_idx = None
    for i in range(1, n):
        if np.isnan(values[i]) or np.isnan(values[i - 1]):
            continue
        if direction == "up" and values[i - 1] <= level < values[i]:
            last_cross_idx = i
        elif direction == "down" and values[i - 1] >= level > values[i]:
            last_cross_idx = i
    if last_cross_idx is None:
        return None
    return (n - 1) - last_cross_idx


def rsi_val_txt_with_bars(v, bars_60, bars_40):
    if pd.isna(v):
        return DOT
    if v > 60:
        return f"{v:.2f} ({bars_60})" if bars_60 is not None else f"{v:.2f}"
    if v < 40:
        return f"{v:.2f} ({bars_40})" if bars_40 is not None else f"{v:.2f}"
    return f"{v:.2f}"


# ─────────────────────────────────────────────────────────────
#  DISPLAY TEXT HELPERS  (match the Pine script's exact symbols)
# ─────────────────────────────────────────────────────────────
def gmma_txt(direction, bars):
    if direction == 0:
        return DOT
    arrow = "▲" if direction == 1 else "▼"
    return f"{arrow} ({int(bars)})"


def wt_tri_txt(direction, bars, cval, ob, os_):
    if direction == 0:
        return DOT
    if direction == 1:
        tri = "▲▲▲" if cval <= os_ else ("▲▲" if cval <= 0 else "▲")
    else:
        tri = "▼▼▼" if cval >= ob else ("▼▼" if cval >= 0 else "▼")
    return f"{tri} ({int(bars)})"


def adx_val_txt(v):
    return DOT if pd.isna(v) else str(int(round(v)))


def di_val_txt(dip, dim):
    if pd.isna(dip) or pd.isna(dim):
        return DOT
    val = dip if dip >= dim else dim
    tri = "▲" if dip >= dim else "▼"
    return f"{int(round(val))}{tri}"


def rsi_val_txt(v):
    return DOT if pd.isna(v) else f"{v:.2f}"


# ─────────────────────────────────────────────────────────────
#  COLORS  (same palette as the Pine script)
# ─────────────────────────────────────────────────────────────
COL_BUY = "#00c853"
COL_SELL = "#d50000"
COL_NEU = "#8a94a6"


def dir_col(direction):
    return COL_BUY if direction == 1 else (COL_SELL if direction == -1 else COL_NEU)


def hc_col(v):
    return COL_BUY if v == 1 else (COL_SELL if v == -1 else COL_NEU)


def di_col(dip, dim):
    if pd.isna(dip) or pd.isna(dim):
        return COL_NEU
    return COL_BUY if dip >= dim else COL_SELL


def rsi_col(v):
    if pd.isna(v):
        return COL_NEU
    if v > 60:
        return COL_BUY
    if v < 40:
        return COL_SELL
    return COL_NEU


# ─────────────────────────────────────────────────────────────
#  ONE-CALL BATCH — everything the table row for a single timeframe needs
# ─────────────────────────────────────────────────────────────
def batch(df: pd.DataFrame, hc_len=10, wt_ch=10, wt_avg=21, wt_ma=4, wt_ob=53, wt_os=-53,
          gmma_smooth=1, gmma_signal=13, adx_len=14, rsi_len=14, intraday=True):
    
    # CRITICAL FIX: Early validation
    if df is None or len(df) < 60:
        return None
    
    # Ensure we have required columns
    required_cols = ['High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        return None
    
    # Drop any rows with NaN in critical columns
    df_clean = df[required_cols].dropna()
    if len(df_clean) < 60:
        return None

    hc_v = h_condition(df_clean, hc_len)
    gmma_dir, gmma_bars = gmma_state(df_clean["Close"], gmma_smooth, gmma_signal)
    wt_dir, wt_bars, wt_cval, wt_ob_v, wt_os_v = wavetrend(df_clean, wt_ch, wt_avg, wt_ma, wt_ob, wt_os)
    adxv, dip, dim = adx_di(df_clean["High"], df_clean["Low"], df_clean["Close"], adx_len)
    rsi_full = rsi_series(df_clean["Close"], rsi_len)
    rsi_v = rsi_full.iloc[-1]
    rsi_bars_60 = rsi_cross_bars(rsi_full, 60, "up") if (not pd.isna(rsi_v) and rsi_v > 60) else None
    rsi_bars_40 = rsi_cross_bars(rsi_full, 40, "down") if (not pd.isna(rsi_v) and rsi_v < 40) else None

    return {
        "hc": hc_v,
        "gmma_dir": gmma_dir, "gmma_bars": gmma_bars,
        "wt_dir": wt_dir, "wt_bars": wt_bars, "wt_cval": wt_cval, "wt_ob": wt_ob_v, "wt_os": wt_os_v,
        "adx": adxv.iloc[-1], "dip": dip.iloc[-1], "dim": dim.iloc[-1],
        "rsi": rsi_v, "rsi_bars_60": rsi_bars_60, "rsi_bars_40": rsi_bars_40,
    }
