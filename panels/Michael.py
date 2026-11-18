
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from datetime import datetime, timedelta
from pathlib import Path
import requests
import io


# settings
TRADING_DAYS = 252
START_DATE = "2015-01-01"
SPLIT_DATE = "2025-01-01"

GOLD_COLOUR = "#D4A017"
SP500_COLOUR = "#2563EB"

DATA_FOLDER = Path("data")
DATA_FOLDER.mkdir(exist_ok=True)

# extra data sources
FRED_DFII10 = "DFII10"
FRED_GVZ = "GVZCLS"

SKEW_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv"
)

# download the extra financial data

@st.cache_data(ttl=3600)
def get_fred_data(series):

    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series}&cosd={START_DATE}"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = pd.read_csv(io.StringIO(response.text))

    data.columns = ["Date", series]
    data["Date"] = pd.to_datetime(data["Date"])
    data[series] = pd.to_numeric(data[series], errors="coerce")

    data = data.dropna(subset=[series])
    data = data.set_index("Date").sort_index()

    # save a copy in the data folder
    data.to_csv(DATA_FOLDER / f"{series}.csv")

    return data


@st.cache_data(ttl=3600)
def get_skew_data():

    response = requests.get(SKEW_URL, timeout=30)
    response.raise_for_status()

    data = pd.read_csv(io.StringIO(response.text))

    date_col = next(
        col for col in data.columns
        if "DATE" in col.upper()
    )

    skew_col = next(
        col for col in data.columns
        if "SKEW" in col.upper()
    )

    data = data[[date_col, skew_col]].copy()
    data.columns = ["Date", "SKEW"]

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["SKEW"] = pd.to_numeric(data["SKEW"], errors="coerce")

    data = data.dropna()
    data = data.set_index("Date").sort_index()

    # save the original observations
    data.to_csv(DATA_FOLDER / "SKEW.csv")

    return data


@st.cache_data(ttl=3600)
def load_extra_data():

    real_rates = get_fred_data(FRED_DFII10)
    gold_volatility = get_fred_data(FRED_GVZ)
    skew = get_skew_data()

    return real_rates, gold_volatility, skew

# load the existing prices and join the new data

from data_loader import get_prices


@st.cache_data(ttl=3600)
def prepare_data():

    # get the shared asset prices
    prices = get_prices()

    # keep gold and the S&P 500
    prices = prices[["Gold", "S&P 500"]].copy()

    # load the three extra datasets
    real_rates, gold_volatility, skew = load_extra_data()

    # join everything using the date
    data = prices.join(real_rates, how="inner")
    data = data.join(gold_volatility, how="inner")
    data = data.join(skew, how="inner")

    # keep the period used in the project
    data = data.loc[START_DATE:]

    # remove any missing observations
    data = data.dropna()

    # make sure dates are in order
    data = data.sort_index()

    return data


def calculate_returns(data):

    # calculate daily log returns for gold and the S&P
    returns = data[["Gold", "S&P 500"]].apply(
        lambda x: np.log(x / x.shift(1))
    )

    returns = returns.dropna()

    # add the extra market information
    returns = returns.join(data[["DFII10", "GVZCLS", "SKEW"]])

    return returns.dropna()
