
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

# --------------------------------------------------
# CHUNK 7: FILTER THE DATA
# --------------------------------------------------

def filter_data(data, period, condition):

    filtered = data.copy()

    # filter by time period
    if period != "All data":
        filtered = filtered[
            filtered["Period"] == period
        ]

    # filter by market condition
    if condition != "All conditions":

        if condition == "High real rates":
            filtered = filtered[
                filtered["Rate regime"] == "High real rates"
            ]

        elif condition == "Low real rates":
            filtered = filtered[
                filtered["Rate regime"] == "Low real rates"
            ]

        elif condition == "High gold volatility":
            filtered = filtered[
                filtered["Gold volatility"] == "High gold volatility"
            ]

        elif condition == "Low gold volatility":
            filtered = filtered[
                filtered["Gold volatility"] == "Low gold volatility"
            ]

        elif condition == "High equity tail risk":
            filtered = filtered[
                filtered["Equity tail risk"] == "High equity tail risk"
            ]

        elif condition == "Low equity tail risk":
            filtered = filtered[
                filtered["Equity tail risk"] == "Low equity tail risk"
            ]

    return filtered


# --------------------------------------------------
# CHUNK 8: CREATE THE SUPPORTING GRAPHS
# --------------------------------------------------

def create_contour_plot(data):

    gold = data["Gold"].to_numpy() * 100
    sp500 = data["S&P 500"].to_numpy() * 100

    values = np.vstack([gold, sp500])
    kde = gaussian_kde(values)

    gold_grid = np.linspace(gold.min(), gold.max(), 60)
    sp500_grid = np.linspace(sp500.min(), sp500.max(), 60)

    X, Y = np.meshgrid(gold_grid, sp500_grid)

    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    figure = go.Figure(
        data=[
            go.Contour(
                x=gold_grid,
                y=sp500_grid,
                z=Z,
                colorscale="Viridis",
                contours=dict(
                    coloring="heatmap",
                    showlabels=True
                )
            )
        ]
    )

    figure.update_layout(
        title="Gold and S&P 500 Return Density",
        xaxis_title="Gold daily return (%)",
        yaxis_title="S&P 500 daily return (%)",
        height=500
    )

    return figure


def create_rolling_correlation(data, window):

    rolling_corr = data["Gold"].rolling(
        window=window
    ).corr(data["S&P 500"])

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr,
            mode="lines",
            name="Rolling correlation",
            line=dict(width=2)
        )
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
        line_color="grey"
    )

    figure.update_layout(
        title=f"Gold and S&P 500 Rolling Correlation ({window} days)",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1, 1]),
        height=450
    )

    return figure


def create_return_distribution(data):

    figure = go.Figure()

    figure.add_trace(
        go.Histogram(
            x=data["Gold"] * 100,
            name="Gold",
            histnorm="probability density",
            opacity=0.65,
            nbinsx=60
        )
    )

    figure.add_trace(
        go.Histogram(
            x=data["S&P 500"] * 100,
            name="S&P 500",
            histnorm="probability density",
            opacity=0.65,
            nbinsx=60
        )
    )

    figure.update_layout(
        title="Daily Return Distributions",
        xaxis_title="Daily return (%)",
        yaxis_title="Density",
        barmode="overlay",
        height=450
    )

    return figure


# --------------------------------------------------
# CHUNK 9: DASHBOARD LAYOUT AND INTERACTIVE CONTROLS
# --------------------------------------------------

st.set_page_config(
    page_title="Gold and Equity Market Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("Gold and Equity Market Analysis")

st.markdown(
    """
    ### Has the relationship between Gold and the S&P 500 changed since 2025?

    Explore how asset returns behave under different economic
    and financial market conditions.
    """
)

# load and prepare the data
try:
    analysis_data = prepare_analysis()

except Exception as error:
    st.error(
        "The data could not be loaded. Check your internet connection, "
        "data sources and data_loader.py."
    )
    st.exception(error)
    st.stop()


# sidebar controls
st.sidebar.title("Dashboard Controls")

period_choice = st.sidebar.selectbox(
    "Select time period",
    [
        "All data",
        "Before 2025",
        "2025 onwards"
    ]
)

condition_choice = st.sidebar.selectbox(
    "Select market condition",
    [
        "All conditions",
        "High real rates",
        "Low real rates",
        "High gold volatility",
        "Low gold volatility",
        "High equity tail risk",
        "Low equity tail risk"
    ]
)

rolling_window = st.sidebar.select_slider(
    "Rolling correlation window",
    options=[30, 60, 90, 180, 252],
    value=90
)

# apply the filters
selected_data = filter_data(
    analysis_data,
    period_choice,
    condition_choice
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"Observations selected: {len(selected_data):,}"
)


# check that enough observations are available
if len(selected_data) < 10:

    st.warning(
        "There are not enough observations for this selection. "
        "Try changing the filters."
    )
    st.stop()


# --------------------------------------------------
# CHUNK 10: DISPLAY THE GRAPHS AND RESULTS
# --------------------------------------------------

# calculate the main statistics
correlation = calculate_correlation(selected_data)

gold_average = selected_data["Gold"].mean() * 100
sp500_average = selected_data["S&P 500"].mean() * 100

gold_volatility = selected_data["Gold"].std() * 100
sp500_volatility = selected_data["S&P 500"].std() * 100


# KPI cards
st.subheader("Market Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Gold–S&P Correlation",
    f"{correlation:.3f}"
)

col2.metric(
    "Gold Average Daily Return",
    f"{gold_average:.3f}%"
)

col3.metric(
    "S&P 500 Average Daily Return",
    f"{sp500_average:.3f}%"
)

col4.metric(
    "Observations",
    f"{len(selected_data):,}"
)


st.markdown("---")


# main 3D graph
st.subheader("1. Joint Probability Density")

st.write(
    """
    This graph estimates how frequently different combinations
    of Gold and S&P 500 daily returns occur.
    """
)

try:

    main_figure = create_3d_surface(
        selected_data,
        "Gold and S&P 500 Joint Probability Density"
    )

    st.plotly_chart(
        main_figure,
        use_container_width=True
    )

except Exception as error:

    st.warning(
        "The 3D density graph could not be calculated for this selection."
    )
    st.caption(str(error))


st.markdown("---")


# compare the two time periods
st.subheader("2. Before and After 2025")

st.write(
    """
    Compare the joint return distributions in the two periods
    to investigate whether the relationship between the assets changed.
    """
)

before_data = analysis_data[
    analysis_data["Period"] == "Before 2025"
]

after_data = analysis_data[
    analysis_data["Period"] == "2025 onwards"
]

comparison_col1, comparison_col2 = st.columns(2)

with comparison_col1:

    st.markdown("**Before 2025**")

    if len(before_data) >= 10:

        try:

            before_figure = create_3d_surface(
                before_data,
                "Before 2025"
            )

            st.plotly_chart(
                before_figure,
                use_container_width=True
            )

        except Exception:

            st.warning(
                "The pre-2025 density graph could not be calculated."
            )

    else:

        st.warning("Not enough pre-2025 observations.")


with comparison_col2:

    st.markdown("**2025 onwards**")

    if len(after_data) >= 10:

        try:

            after_figure = create_3d_surface(
                after_data,
                "2025 onwards"
            )

            st.plotly_chart(
                after_figure,
                use_container_width=True
            )

        except Exception:

            st.warning(
                "The post-2025 density graph could not be calculated."
            )

    else:

        st.warning("Not enough post-2025 observations.")


# correlation comparison
st.subheader("Correlation Comparison")

comparison_results = []

for period_name, period_data in [
    ("Before 2025", before_data),
    ("2025 onwards", after_data)
]:

    if len(period_data) >= 2:

        comparison_results.append({
            "Period": period_name,
            "Correlation": period_data["Gold"].corr(
                period_data["S&P 500"]
            ),
            "Gold volatility (%)": period_data["Gold"].std() * 100,
            "S&P 500 volatility (%)": period_data["S&P 500"].std() * 100,
            "Observations": len(period_data)
        })

comparison_table = pd.DataFrame(comparison_results)

st.dataframe(
    comparison_table,
    use_container_width=True,
    hide_index=True
)


st.markdown("---")


# 2D density contour graph
st.subheader("3. Two-Dimensional Density Contours")

st.write(
    """
    The contour graph provides another way to see where combinations
    of Gold and S&P 500 returns are concentrated.
    """
)

try:

    contour_figure = create_contour_plot(selected_data)

    st.plotly_chart(
        contour_figure,
        use_container_width=True
    )

except Exception as error:

    st.warning(
        "The density contour graph could not be calculated."
    )
    st.caption(str(error))


st.markdown("---")


# rolling correlation graph
st.subheader("4. Rolling Correlation")

st.write(
    """
    This graph tracks how the relationship between Gold and the
    S&P 500 changes over time.
    """
)

rolling_figure = create_rolling_correlation(
    analysis_data,
    rolling_window
)

st.plotly_chart(
    rolling_figure,
    use_container_width=True
)


st.markdown("---")


# return distribution graph
st.subheader("5. Daily Return Distributions")

st.write(
    """
    Compare the distribution of daily Gold and S&P 500 returns.
    """
)

distribution_figure = create_return_distribution(
    selected_data
)

st.plotly_chart(
    distribution_figure,
    use_container_width=True
)


st.markdown("---")


# summary statistics
st.subheader("6. Summary Statistics")

summary_table = calculate_summary(selected_data)

st.dataframe(
    summary_table,
    use_container_width=True
)


# display the underlying data
with st.expander("View underlying dataset"):

    st.dataframe(
        selected_data,
        use_container_width=True
    )


# project information
st.markdown("---")

st.caption(
    """
    Data sources: Yahoo Finance, Federal Reserve Economic Data (FRED)
    and CBOE. Daily returns are calculated from historical prices.
    Market regimes are classified using the median of the available sample.
    """
)
