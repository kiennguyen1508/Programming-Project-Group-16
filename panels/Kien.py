"""
Panel 1: Performance and ratios (Kien).
"""
import streamlit as st

from data_loader import get_prices, SPLIT

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
    st.dataframe(recent.tail())