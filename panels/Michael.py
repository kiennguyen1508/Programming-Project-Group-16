
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

# calculate the different market conditions

def add_market_regimes(data):

    data = data.copy()

    # use the median to separate high and low conditions
    rate_level = data["DFII10"].median()
    gold_vol_level = data["GVZCLS"].median()
    skew_level = data["SKEW"].median()

    data["Rate regime"] = np.where(
        data["DFII10"] >= rate_level,
        "High real rates",
        "Low real rates"
    )

    data["Gold volatility"] = np.where(
        data["GVZCLS"] >= gold_vol_level,
        "High gold volatility",
        "Low gold volatility"
    )

    data["Equity tail risk"] = np.where(
        data["SKEW"] >= skew_level,
        "High equity tail risk",
        "Low equity tail risk"
    )

    return data


def prepare_analysis():

    # load the combined dataset
    data = prepare_data()

    # calculate daily returns
    returns = calculate_returns(data)

    # add the market regime labels
    returns = add_market_regimes(returns)

    # mark observations before and during 2025
    returns["Period"] = np.where(
        returns.index >= SPLIT_DATE,
        "2025 onwards",
        "Before 2025"
    )

    return returns

# calculate the joint probability density for gold and the S&P 500

def calculate_kde(data, grid_size=60):

    # get the two daily return columns
    gold = data["Gold"].to_numpy() * 100
    sp500 = data["S&P 500"].to_numpy() * 100

    # combine the returns into one array
    values = np.vstack([gold, sp500])

    # estimate the joint probability density
    kde = gaussian_kde(values)

    # create a grid of possible return combinations
    gold_grid = np.linspace(gold.min(), gold.max(), grid_size)
    sp500_grid = np.linspace(sp500.min(), sp500.max(), grid_size)

    X, Y = np.meshgrid(gold_grid, sp500_grid)

    # calculate density at every point on the grid
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    return X, Y, Z


def create_3d_surface(data, title):

    X, Y, Z = calculate_kde(data)

    # create the 3D surface
    figure = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="Viridis",
                colorbar=dict(title="Density")
            )
        ]
    )

    figure.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Gold daily return (%)",
            yaxis_title="S&P 500 daily return (%)",
            zaxis_title="Joint probability density"
        ),
        height=650
    )

    return figure

# compare the gold and S&P 500 distributions across periods

def compare_periods(data):

    # separate the observations into two periods
    before_2025 = data[
        data["Period"] == "Before 2025"
    ]

    after_2025 = data[
        data["Period"] == "2025 onwards"
    ]

    # create the two 3D density graphs
    figure_before = create_3d_surface(
        before_2025,
        "Gold and S&P 500 Joint Distribution: Before 2025"
    )

    figure_after = create_3d_surface(
        after_2025,
        "Gold and S&P 500 Joint Distribution: 2025 Onwards"
    )

    return figure_before, figure_after


# calculate the correlation between Gold and the S&P 500

def calculate_correlation(data):

    correlation = data["Gold"].corr(data["S&P 500"])

    return correlation


# calculate average daily returns and volatility

def calculate_summary(data):

    summary = pd.DataFrame({
        "Average daily return (%)": [
            data["Gold"].mean() * 100,
            data["S&P 500"].mean() * 100
        ],
        "Daily volatility (%)": [
            data["Gold"].std() * 100,
            data["S&P 500"].std() * 100
        ]
    }, index=["Gold", "S&P 500"])

    return summary
