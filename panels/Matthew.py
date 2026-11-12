"""Panel 3: Correlation Heatmap - Matthew"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import get_returns, SPLIT    


def split_periods(returns: pd.DataFrame, assets: list) -> tuple:
    """Split daily returns into 'before 2025' and 'after 2025'."""
    chosen = returns[assets] #keep only the chosen assets

    before = chosen[chosen.index < SPLIT]  #rows dated before 1 Jan 2025

    after = chosen[chosen.index >= SPLIT]  #rows dated on or after 1 Jan 2025

    return before, after

def correlation_table(returns: pd.DataFrame) -> tuple:
    """Return the correlation (r) and p-value for every pair of assets"""
    r_table = returns.corr()              # r for every pair of assets
    p_table = r_table.copy()              # same_shaped table, to fill with p values in the heatmap
    for a in returns.columns:
        for b in returns.columns:
            p_table.loc[a,b] = stats.pearsonr(returns[a], returns[b]).pvalue
            return r_table, p_table

def render():
    """Draw Panel 3 on the dashboard"""
    st.subheader("Panel 3: Correlation Heatmap")
    returns = get_returns()
    before, since = split_periods(returns, list(returns.columns))
    st.write(f"Days before 2025: {len(before)} | since 2025: {len(since)}")
    r_before, p_before = correlation_table(before)
    r_since, p_since = correlation_table(since)
    st.write("Correlation of daily returns, 2015-2024")
    st.dataframe(r_before.round(2))
    st.write("Correlation of daily returns, since Jan 2025")
    st.dataframe(r_since.round(2))

                        