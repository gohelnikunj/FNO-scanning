"""
stock_list.py
--------------
Single source of truth for the stock universe. Both streamlit_app.py (the
live dashboard) and update_data.py (the GitHub Actions background updater)
import STOCKS from here, so they can never drift out of sync with each other.

Edit this list freely to add/remove stocks — both the dashboard and the
background updater will pick up the change automatically.
"""

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
    {"name": "Tata Motors Passenger Vehicles", "ticker": "TMPV.NS", "sector": "Auto", "tier": 1,
     "note": "Formerly TATAMOTORS — renamed after Oct 2025 demerger (PV + JLR business)"},
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
