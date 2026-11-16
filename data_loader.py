"""
Shared data for our group dashboard.

Run once to download: python data_loader.py
Then in any panel: from data_loader import get_prices, get_returns, SPLIT
"""

from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = {"Gold": "GC=F", "Silver": "SI=F", "Oil": "BZ=F", "S&P 500": "^GSPC"}
START = "2015-01-01"
END = "2026-11-01" # Since yfinance is not always up to date, we set the end date to a future date to ensure we get the latest data.
SPLIT = "2025-01-01" # "before" vs "since 2025"

DATA_DIR = Path(__file__).parent / "data"
CSV = DATA_DIR / "prices.csv"

def download():
    """
    Download daily closing prices from Yahoo Finance and save them to data/prices.csv.
    """
    import yfinance as yf

    raw = yf.download(list(TICKERS.values()), start = START , end = END, 
                      auto_adjust = True , progress = False ) ["Close"]

    prices = raw.rename(columns = {v: k for k , v in TICKERS.items()})
    prices = prices[list(TICKERS)] # fixed column order
    prices = prices.dropna() # keep only days when all four traded
    if prices.empty:
        raise ValueError("No data download. Check the tickers and check your internet connect")
    prices.index.name = "Date"

    DATA_DIR.mkdir(exist_ok = True)
    prices.round(4).to_csv(CSV)
    print(f"Saved {len(prices)} rows to {CSV}")
    print(f"From {prices.index[0].date()} to {prices.index[-1].date()}")
    print(prices.tail())

def get_prices(start = None , end = None):
    """
    Daily closing prices from the saved CSV, one column per asset.
    """
    prices = pd.read_csv(CSV , index_col="Date" , parse_dates = True)
    return prices.loc[start : end]

def get_returns(start = None , end = None):
    """
    Daily log returns: the same definition for every panel.
    """
    return np.log(get_prices(start , end)).diff().dropna()

# ---------- Extra data: what drives gold (kept separate from prices.csv) ----------
MACRO_TICKERS = {"Dollar":"DX-Y.NYB" , "Copper":"HG=F" , "Commodities":"DBC"}
FRED_SERIES = {"Real yield":"DFII10"} # 10-year TIPS yield, in%
VIX_CSV = DATA_DIR / "VIX_History.csv"
MACRO_CSV = DATA_DIR / "macro.csv"

def download_macro():
    """
    Download the dollar index, copper, a broad commodity ETF and the real yield,
    add the VIX from the saved CBOE file, and save everything to data/macro.csv.
    """
    import io
    import requests
    import yfinance as yf

    raw = yf.download(list(MACRO_TICKERS.values()), start=START, end=END,
                      auto_adjust=True, progress=False)["Close"]
    macro = raw.rename(columns={v:k for k, v in MACRO_TICKERS.items()})

    for name, code in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        fred = pd.read_csv(io.StringIO(response.text), index_col=0,
                            parse_dates=True, na_values=".")
        macro[name] = fred.iloc[:, 0]

    vix = pd.read_csv(VIX_CSV, index_col="DATE", parse_dates=True,
                      date_format="%m/%d/%Y")["CLOSE"]
    macro["VIX"] = vix

    macro.index.name="Date"
    macro.round(4).to_csv(MACRO_CSV)
    print(f"Saved {len(macro)} rows to {MACRO_CSV}")
    print(macro.isna().sum())

def get_macro(start=None, end=None):
    """
    Levels of the extra series: prices for Dollar, Copper and Commodities,
    % for Real yield, index points for VIX
    """
    macro = pd.read_csv(MACRO_CSV, index_col="Date", parse_dates=True)
    return macro.loc[start:end]

def get_macro_changes(start=None, end=None):
    """
    Daily changes: log returns for prices, basis points for the real yield,
    points for the VIX (yields and VIX are levels, so log returns don't make sense).
    """
    macro = get_macro(start,end)
    changes = pd.DataFrame(index=macro.index)
    for col in MACRO_TICKERS:
        changes[col] = np.log(macro[col]).diff()
    changes["Real yield"] = macro["Real yield"].diff()*100
    changes["VIX"] = macro["VIX"].diff()
    return changes


if __name__ == "__main__":
    import sys
    if "macro" in sys.argv:
        download_macro()
    else:
        download()