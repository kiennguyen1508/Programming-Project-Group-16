"""
Panel 1: Performance and ratios (Kien).
"""
import streamlit as st
import plotly.graph_objects as go

from data_loader import get_prices, SPLIT


COLOURS = {"Gold": "#B45309", "Silver": "#7C8798", "Oil": "#0F766E", "S&P 500": "#2563EB"}

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
    recent = prices.loc[SPLIT:]

    st.caption(
        f"Daily closes from {recent.index[0].date()} to {recent.index[-1].date()} "
        f"({len(recent)} trading days)"
    )
    rebased = recent / recent.iloc[0] * 100
    st.plotly_chart(make_price_chart(rebased))

def make_price_chart(rebased):
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
    return fig