"""
update_data.py
----------------
Runs on a schedule via GitHub Actions (see .github/workflows/update_data.yml)
— on GitHub's own servers, completely independent of anyone having the
dashboard open in a browser. Fetches fresh OHLCV data from Yahoo Finance for
every stock in stock_list.py, computes the same indicators the live
dashboard uses, and writes everything to data/latest.json.

streamlit_app.py reads that file instead of calling Yahoo Finance itself —
so the dashboard shows genuinely current data even if nobody has visited
it in hours, as long as this job has run recently.

Run manually for testing: python update_data.py
"""

import json
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import yfinance as yf

import indicators as ind
from stock_list import STOCKS

IST = timezone(timedelta(hours=5, minutes=30))

TF_SPECS = {
    "Day": ("1d", "1y", False),
    "1H":  ("1h", "1mo", True),
    "5M":  ("5m", "5d", True),
}


def sanitize(v):
    """Convert numpy scalars to plain Python types so json.dump can handle them."""
    if isinstance(v, dict):
        return {k: sanitize(x) for k, x in v.items()}
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def fetch_one_timeframe(ticker: str, interval: str, period: str):
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
            time.sleep(1.5)
    return None, last_err


def fetch_and_compute(ticker: str):
    tech = {}
    errors = {}
    ok = True
    for label, (interval, period, intraday) in TF_SPECS.items():
        df, err = fetch_one_timeframe(ticker, interval, period)
        result = ind.batch(df, intraday=intraday) if df is not None else None
        tech[label] = sanitize(result) if result else None
        if result is None:
            ok = False
            errors[label] = err or "unknown error"
    return tech, ok, errors


def is_market_hours(now_ist: datetime) -> bool:
    """Mon-Fri, ~8:15 AM - 4:35 PM IST (a little buffer either side of the
    real 9:15-15:30 session). Returns False on weekends / outside this
    window, so it's safe to trigger this script from an external cron
    service on a simple fixed interval without that service needing to
    know about market hours itself — this script just no-ops otherwise.
    Set ALLOW_OFF_HOURS=1 as an env var to bypass this (useful for testing).
    """
    import os
    if os.environ.get("ALLOW_OFF_HOURS") == "1":
        return True
    if now_ist.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    minutes = now_ist.hour * 60 + now_ist.minute
    return (8 * 60 + 15) <= minutes <= (16 * 60 + 35)


def main():
    started = datetime.now(IST)

    if not is_market_hours(started):
        print(f"Outside market hours ({started.strftime('%a %H:%M IST')}) — skipping this run, "
              f"data/latest.json left untouched.")
        return

    out = {"generated_at": started.isoformat(), "stocks": {}}
    failed_count = 0

    for i, s in enumerate(STOCKS):
        ticker = s["ticker"]
        tech, ok, errors = fetch_and_compute(ticker)
        out["stocks"][ticker] = {"tech": tech, "ok": ok, "errors": errors}
        if not ok:
            failed_count += 1
        print(f"[{i+1}/{len(STOCKS)}] {ticker}: {'OK' if ok else 'FAILED — ' + str(errors)}")
        time.sleep(0.4)  # pacing between stocks to avoid Yahoo rate limits

    out["finished_at"] = datetime.now(IST).isoformat()
    out["failed_count"] = failed_count
    out["total_count"] = len(STOCKS)

    with open("data/latest.json", "w") as f:
        json.dump(out, f, indent=2, allow_nan=True)

    print(f"\nDone. Wrote data/latest.json — {len(STOCKS) - failed_count}/{len(STOCKS)} stocks OK.")


if __name__ == "__main__":
    main()
