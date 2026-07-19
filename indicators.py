"""
indicators.py
--------------
A faithful Python port of the Pine Script's indicator functions, so the
Streamlit dashboard can reproduce the exact same 9-column table
(TF | H | GMMA | WT | STCR | ADX | DI | RSI | SF) per stock, per timeframe.

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
#  STOCHASTIC RSI
# ─────────────────────────────────────────────────────────────
def stoch_rsi(df: pd.DataFrame, rsi_len: int, stoch_len: int, k_smooth: int, d_smooth: int):
    rsi_v = rsi_series(df["Close"], rsi_len)
    lo = rsi_v.rolling(stoch_len).min()
    hi = rsi_v.rolling(stoch_len).max()
    k_raw = 100 * (rsi_v - lo) / (hi - lo).replace(0, np.nan)
    k_line = sma(k_raw, k_smooth)
    d_line = sma(k_line, d_smooth)
    direction, bars = cross_state(k_line, d_line)
    kval = k_line.iloc[-1]
    return direction, bars, kval


# ─────────────────────────────────────────────────────────────
#  SF 4-FACTOR
#  APPROX: Pine's ta.vwap resets every session. For 5M/1H we rebuild that by
#  resetting the cumulative VWAP at each new calendar day. For Daily bars,
#  a "session" *is* the bar, so we approximate session VWAP as that day's
#  own typical price (hlc3) rather than a multi-day VWAP.
# ─────────────────────────────────────────────────────────────
def session_vwap(df: pd.DataFrame, intraday: bool) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    if not intraday:
        return typical  # APPROX — see note above
    vol = df["Volume"].replace(0, np.nan).ffill().fillna(1)
    day = df.index.date
    pv = typical * vol
    pv_cum = pv.groupby(day).cumsum()
    vol_cum = vol.groupby(day).cumsum()
    return pv_cum / vol_cum


def calc_sf(df: pd.DataFrame, rsi_fast=25, rsi_slow=55, pvt_sig=21, atr_len=22, atr_mul=3.0, intraday=True) -> int:
    src = df["Close"]
    rsi_f = rsi_series(src, rsi_fast)
    rsi_s = rsi_series(src, rsi_slow)

    change_pct = src.pct_change().fillna(0)
    pvt = (change_pct * df["Volume"]).cumsum()
    pvt_s = sma(pvt, pvt_sig)

    vwap_v = session_vwap(df, intraday)
    atr_v = atr_mul * atr_series(df["High"], df["Low"], df["Close"], atr_len)

    highest_src = src.rolling(atr_len).max()
    lowest_src = src.rolling(atr_len).min()
    new_long = (highest_src - atr_v).values
    new_short = (lowest_src + atr_v).values
    src_v = src.values

    n = len(src_v)
    long_stop = np.zeros(n)
    short_stop = np.zeros(n)
    sf_dir = np.ones(n, dtype=int)

    for i in range(n):
        if i == 0 or np.isnan(new_long[i]):
            long_stop[i] = new_long[i] if not np.isnan(new_long[i]) else 0.0
            short_stop[i] = new_short[i] if not np.isnan(new_short[i]) else 0.0
            continue
        long_stop[i] = max(new_long[i], long_stop[i - 1]) if src_v[i - 1] > long_stop[i - 1] else new_long[i]
        short_stop[i] = min(new_short[i], short_stop[i - 1]) if src_v[i - 1] < short_stop[i - 1] else new_short[i]

        if src_v[i] > short_stop[i - 1]:
            sf_dir[i] = 1
        elif src_v[i] < long_stop[i - 1]:
            sf_dir[i] = -1
        else:
            sf_dir[i] = sf_dir[i - 1]

    i = n - 1
    buy_sig = (
        sf_dir[i] == 1
        and rsi_f.iloc[i] > rsi_s.iloc[i]
        and pvt.iloc[i] > pvt_s.iloc[i]
        and src.iloc[i] > vwap_v.iloc[i]
    )
    sell_sig = (
        sf_dir[i] == -1
        and rsi_s.iloc[i] > rsi_f.iloc[i]
        and pvt.iloc[i] < pvt_s.iloc[i]
        and src.iloc[i] < vwap_v.iloc[i]
    )
    if buy_sig:
        return 1
    if sell_sig:
        return -1
    return 0


# ─────────────────────────────────────────────────────────────
#  RSI THRESHOLD-CROSS BAR COUNTER
#  "n" = candles since RSI last crossed the given level in the given
#  direction — same style as the GMMA/WT/STCR "bars since cross" cells.
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


def stcr_tri_txt(direction, bars, kval):
    if direction == 0:
        return DOT
    if direction == 1:
        tri = "▲▲▲" if kval < 20 else ("▲▲" if kval <= 80 else "▲")
    else:
        tri = "▼▼▼" if kval > 80 else ("▼▼" if kval >= 20 else "▼")
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


def sf_txt(v):
    return "BUY" if v == 1 else ("SELL" if v == -1 else "NEU")


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


def sf_col(v):
    return COL_BUY if v == 1 else (COL_SELL if v == -1 else COL_NEU)


# ─────────────────────────────────────────────────────────────
#  ONE-CALL BATCH — everything the table row for a single timeframe needs
# ─────────────────────────────────────────────────────────────
def batch(df: pd.DataFrame, hc_len=10, wt_ch=10, wt_avg=21, wt_ma=4, wt_ob=53, wt_os=-53,
          gmma_smooth=1, gmma_signal=13, adx_len=14,
          stcr_rsi_len=14, stcr_stoch_len=14, stcr_k=3, stcr_d=3,
          rsi_len=14, sf_rsi_fast=25, sf_rsi_slow=55, sf_pvt_sig=21, sf_atr_len=22, sf_atr_mul=3.0,
          intraday=True):
    if df is None or len(df) < max(60, sf_atr_len + 2):
        return None

    hc_v = h_condition(df, hc_len)
    gmma_dir, gmma_bars = gmma_state(df["Close"], gmma_smooth, gmma_signal)
    wt_dir, wt_bars, wt_cval, wt_ob_v, wt_os_v = wavetrend(df, wt_ch, wt_avg, wt_ma, wt_ob, wt_os)
    stcr_dir, stcr_bars, stcr_kv = stoch_rsi(df, stcr_rsi_len, stcr_stoch_len, stcr_k, stcr_d)
    adxv, dip, dim = adx_di(df["High"], df["Low"], df["Close"], adx_len)
    rsi_full = rsi_series(df["Close"], rsi_len)
    rsi_v = rsi_full.iloc[-1]
    rsi_bars_60 = rsi_cross_bars(rsi_full, 60, "up") if (not pd.isna(rsi_v) and rsi_v > 60) else None
    rsi_bars_40 = rsi_cross_bars(rsi_full, 40, "down") if (not pd.isna(rsi_v) and rsi_v < 40) else None
    sf_v = calc_sf(df, sf_rsi_fast, sf_rsi_slow, sf_pvt_sig, sf_atr_len, sf_atr_mul, intraday)

    return {
        "hc": hc_v,
        "gmma_dir": gmma_dir, "gmma_bars": gmma_bars,
        "wt_dir": wt_dir, "wt_bars": wt_bars, "wt_cval": wt_cval, "wt_ob": wt_ob_v, "wt_os": wt_os_v,
        "stcr_dir": stcr_dir, "stcr_bars": stcr_bars, "stcr_kv": stcr_kv,
        "adx": adxv.iloc[-1], "dip": dip.iloc[-1], "dim": dim.iloc[-1],
        "rsi": rsi_v, "rsi_bars_60": rsi_bars_60, "rsi_bars_40": rsi_bars_40,
        "sf": sf_v,
    }
