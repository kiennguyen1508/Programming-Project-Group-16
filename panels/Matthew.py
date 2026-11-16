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

def load_macro() -> pd.DataFrame:
    """Load the macro data and turn each column into a daily change."""
    macro = pd.read_csv("data/macro.csv", index_col="Date", parse_dates=True)
    changes = macro.pct_change()                          # daily % change of each level
    changes["Real yield"] = macro["Real yield"].diff()    # yields: change in % points instead
    return changes

def correlation_table(returns: pd.DataFrame) -> tuple:
    """Return the correlation (r) and p-value for every pair of assets"""
    r_table = returns.corr()              # r for every pair of assets
    p_table = r_table.copy()              # same_shaped table, to fill with p values in the heatmap
    for a in returns.columns:
        for b in returns.columns:
            p_table.loc[a,b] = stats.pearsonr(returns[a], returns[b]).pvalue
    return r_table, p_table

def make_heatmap(r_table: pd.DataFrame, title: str):
    """Draw one correlation heatmap: red=move together, blue = move opposite."""
    fig = px.imshow(r_table.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title = title, )
    return fig 

def hide_cells(r_table: pd.DataFrame, p_table: pd.DataFrame, cutoff: float = 0.05):
    """Blank out the repeated half, and other non statisitically significant cells."""
    bottom_half = np.tril(np.ones(r_table.shape, dtype=bool))
    not_significant = p_table.values >= cutoff
    hidden = r_table.mask(bottom_half | not_significant)
    return hidden.iloc[:-1,1:]

def fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> tuple:
    """Test whether two correlations from separate periods are really different."""
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1 / (n1-3) + 1 / (n2 - 3))
    z =(z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def render():
    """Draw Panel 3 on the dashboard"""
    st.subheader("Panel 3: Correlation Heatmap")
    returns = get_returns().join(load_macro(), how="inner").dropna()
    assets = st.multiselect("Assets", list(returns.columns),
                        default=["Gold", "Silver", "Oil", "S&P 500", "Real yield"],
                        key="p3_assets")
    cutoff = st.select_slider("Significance level (p-value cutoff)",
                        options=[0.01, 0.05, 0.10], value=0.05, key="p3_cutoff")
    if len(assets) < 2:
        st.warning("Pick at least two assets to compare.")
        return
    before, since = split_periods(returns, assets)
    st.write(f"Days before 2025: {len(before)} | since 2025: {len(since)}")
    r_before, p_before = correlation_table(before)
    r_since, p_since = correlation_table(since)
    left, right = st.columns(2)     # split the space into two side by side boxes
    left.plotly_chart(make_heatmap(hide_cells(r_before, p_before, cutoff),"2015-2024"))
    right.plotly_chart(make_heatmap(hide_cells(r_since, p_since, cutoff),"Since Jan 2025"))
    st.caption("Blank cells are statistically insignificant.")
    a, b = "Gold", "Oil"
    z, p = fisher_z_test(r_before.loc[a, b], len(before), r_since.loc[a, b], len(since))
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{a}-{b} 2015-2024", f"r = {r_before.loc[a, b]:+.2f}")
    c2.metric(f"{a}-{b} since 2025", f"r = {r_since.loc[a, b]:+.2f}")
    c3.metric("Fisher z-test", f"z = {z:.2f}", f"p = {p:.3f}", delta_color="off")
                        