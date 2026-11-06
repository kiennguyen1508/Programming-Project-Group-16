"""
Panel 1: Performance and ratios (Kien).
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data_loader import get_prices, SPLIT


COLOURS = {"Gold": "#B45309", "Silver": "#7C8798", "Oil": "#0F766E", "S&P 500": "#2563EB"}
RATIOS = {
    "Gold / Silver" : ("Gold", "Silver"),
    "Gold / Oil" : ("Gold" , "Oil"),
}

@st.cache_data
def load_prices():
    """
    Load the shared price data once, then reuse it on every rerun.
    """
    return get_prices()

def render():
    """
    Draw Panel 1 inside the dashboard.
    """
    st.subheader("Performance & ratios")

    prices = load_prices()

    view = st.radio("View" , ["Prices" , "Ratios"] , horizontal = True , key = "p1_view")
    period = st.radio(
        "Period",
        ["Since 2025" , "YTD" , "1Y" , "5Y" , "Max"],
        horizontal = True,
        key = "p1_period",
    )
    recent = prices.loc[start_date(prices, period):]

    st.caption(
        f"Daily closes from {recent.index[0].date()} to {recent.index[-1].date()} "
        f"({len(recent)} trading days)"
    )
    if view == "Prices":
        selected = st.multiselect(
            "Assets",
            options = list(COLOURS),
            default = list(COLOURS),
            key = "p1_assets",
        )
        log_scale = st.toggle("Log scale" , key = "p1_log")
        chosen = recent[selected]
        rebased = chosen / chosen.iloc[0] * 100
        st.plotly_chart(make_price_chart(rebased , log_scale))

    else:
        name = st.radio("Ratio" , list(RATIOS) , horizontal = True , key = "p1_ratio")
        top, bottom = RATIOS[name]
        ratio = recent[top] / recent[bottom]
        full = prices[top] / prices[bottom]
        mean, std = full.mean(), full.std()
        st.plotly_chart(make_ratio_chart(ratio, name, mean, std))

def start_date(prices, period):
    """
    Turn a period label into the first date to show.
    """
    last = prices.index[-1]
    if period == "Since 2025":
        return pd.Timestamp(SPLIT)
    if period == "YTD":
        return pd.Timestamp(year = last.year , month = 1 , day = 1)
    if period == "1Y":
        return last - pd.DateOffset(years = 1)
    if period == "5Y":
        return last - pd.DateOffset(years = 5)
    return prices.index[0] #"Max"


def make_price_chart(rebased, log_scale = False):
    """
    Line chart of each asset, rebased so every line starts at 100.
    """
    fig = go.Figure()
    for asset in rebased.columns:
        fig.add_trace(go.Scatter(
            x = rebased.index,
            y = rebased[asset],
            name = asset,
            mode = "lines",
            line = dict(color = COLOURS[asset] , width = 2),
            hovertemplate = "%{y:.1f}",
        ))
    fig.add_hline(y = 100 , line_dash = "dot" , line_color = "grey")
    fig.update_layout(
        height = 420,
        hovermode = "x unified",
        yaxis_title = "Value (start = 100)",
        legend = dict(orientation = "h" , y = 1.1),
        margin = dict(l=10 , r=10 , t=30 , b=10),
    )
    if log_scale:
        fig.update_yaxes(type = "log")
    return fig

def make_ratio_chart(ratio, name, mean, std):
    """
    Line chart of one price ratio, with its long-run mean and ±1σ/±2σ bands. 
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = ratio.index,
        y = ratio,
        name = name,
        mode = "lines",
        line = dict(color = COLOURS["Gold"] , width = 2),
        hovertemplate = "%{y:.1f}",
    ))
    fig.add_hline(y = mean, line_color = "grey" , annotation_text = "Mean",
                  annotation_position = "right")
    for k, dash in [(1 , "dash") , (2 , "dot")]:
        fig.add_hline(y=mean + k * std, line_dash = dash, line_color = "grey",
                      annotation_text = f"+{k}σ ", annotation_position = "right")
        fig.add_hline(y=mean - k * std, line_dash = dash , line_color = "grey",
                      annotation_text = f"-{k}σ " , annotation_position = "right")

    fig.update_layout(
        height = 420,
        hovermode = "x unified",
        yaxis_title = f"{name} ratio",
        showlegend = False,
        margin = dict (l=10, r=50, t=30, b=10),
    )
    return fig