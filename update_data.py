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
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import yfinance as yf

import indicators as ind
from stock_list import STOCKS

IST = timezone(timedelta(hours=5, minutes=30))

TF_SPECS = {
    "Day": ("1d", "2y", False),    # Increased from 1y to 2y for more data
    "1H":  ("1h", "3mo", True),     # Increased from 1mo to 3mo
    "5M":  ("5m", "1mo", True),     # Increased from 5d to 1mo
}

def sanitize(v):
    """Convert numpy scalars to plain Python types so json.dump can handle them."""
    if isinstance(v, dict):
        return {k: sanitize(x) for k, x in v.items()}
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (pd.Series, pd.DataFrame)):
        return v.to_dict()
    return v

def fetch_one_timeframe(ticker: str, interval: str, period: str):
    """Fetch with retries and longer timeout"""
    last_err = None
    for attempt in range(3):  # Increased retries
        try:
            # Add timeout and more robust fetching
            df = yf.download(
                ticker, 
                period=period, 
                interval=interval,
                progress=False,
                timeout=30,
                auto_adjust=False
            )
            if df is not None and not df.empty and len(df) >= 10:
                return df, None
            elif df is not None and not df.empty:
                last_err = f"Only {len(df)} bars returned (need 60 for indicators)"
            else:
                last_err = "Yahoo returned no data (empty response)"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        
        if attempt < 2:
            wait_time = 2 ** attempt  # 1s, 2s, 4s backoff
            print(f"  Retry {attempt+1}/3 in {wait_time}s...")
            time.sleep(wait_time)
    
    return None, last_err

def fetch_and_compute(ticker: str):
    tech = {}
    errors = {}
    ok = True
    
    print(f"\n📊 Processing {ticker}...")
    
    for label, (interval, period, intraday) in TF_SPECS.items():
        print(f"  Fetching {label} ({interval}, {period})...")
        df, err = fetch_one_timeframe(ticker, interval, period)
        
        if df is not None and len(df) >= 60:  # Need minimum 60 bars
            result = ind.batch(df, intraday=intraday)
            tech[label] = sanitize(result) if result else None
            if result is None:
                ok = False
                errors[label] = "Indicator computation failed (insufficient data?)"
                print(f"    ❌ {label}: Indicator computation failed")
            else:
                print(f"    ✅ {label}: {len(df)} bars, indicators computed")
        else:
            ok = False
            error_msg = err or f"Only {len(df) if df is not None else 0} bars"
            errors[label] = error_msg
            tech[label] = None
            print(f"    ❌ {label}: {error_msg}")
    
    return tech, ok, errors

def is_market_hours(now_ist: datetime) -> bool:
    """Wider market hours for data availability"""
    import os
    if os.environ.get("ALLOW_OFF_HOURS") == "1":
        print("⚠ ALLOW_OFF_HOURS=1 — running regardless of market hours")
        return True
    
    # Weekends
    if now_ist.weekday() >= 5:
        return False
    
    # Wider window: 7 AM - 6 PM IST (covers pre/post market data)
    minutes = now_ist.hour * 60 + now_ist.minute
    return (7 * 60) <= minutes <= (18 * 60)

def main():
    started = datetime.now(IST)
    print(f"\n{'='*60}")
    print(f"🚀 UPDATE STARTED: {started.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"{'='*60}\n")
    
    if not is_market_hours(started):
        print(f"⏰ Outside market hours ({started.strftime('%a %H:%M IST')})")
        print("   Data/latest.json left untouched")
        return
    
    out = {
        "generated_at": started.isoformat(),
        "finished_at": None,
        "stocks": {},
        "failed_count": 0,
        "total_count": len(STOCKS)
    }
    failed_count = 0
    
    for i, s in enumerate(STOCKS):
        ticker = s["ticker"]
        tech, ok, errors = fetch_and_compute(ticker)
        out["stocks"][ticker] = {
            "tech": tech, 
            "ok": ok, 
            "errors": errors,
            "last_fetch": datetime.now(IST).isoformat()
        }
        if not ok:
            failed_count += 1
        
        # Progress with backoff between stocks
        print(f"\n  [{i+1}/{len(STOCKS)}] {ticker}: {'✅ OK' if ok else '❌ FAILED'}")
        if not ok:
            print(f"    Errors: {errors}")
        
        # Longer delay between stocks to avoid rate limits
        if i < len(STOCKS) - 1:
            time.sleep(1.0)
    
    out["finished_at"] = datetime.now(IST).isoformat()
    out["failed_count"] = failed_count
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    with open("data/latest.json", "w") as f:
        json.dump(out, f, indent=2, allow_nan=True)
    
    print(f"\n{'='*60}")
    print(f"✅ DONE: {len(STOCKS) - failed_count}/{len(STOCKS)} stocks OK")
    print(f"   Failed: {failed_count}")
    print(f"   Data saved to data/latest.json")
    print(f"   Finished: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
