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
    st.subheader("Correlation Heatmap")
    returns = get_returns().join(load_macro(), how="inner").dropna()

    # 1. Controls
    assets = st.multiselect("Assets", list(returns.columns),
                        default=["Gold", "Silver", "Oil", "S&P 500", "Real yield"],
                        key="p3_assets")
    first, last = returns.index.min().date(), returns.index.max().date()
    start, end = st.slider("Time Period", min_value=first, max_value=last,
                           value=(pd.Timestamp(SPLIT).date(), last),
                           format="MMM YYYY", key="p3_dates")
    cutoff = st.slider("Significance level (p-value cutoff)", min_value=0.01,
                       max_value=0.20, value=0.05, step=0.01, format="%.2f", key="p3_cutoff")
                        

    # 2. Checking the choices make sense.
    period = returns.loc[str(start):str(end)]
    if len(assets) < 2 or len(period) < 30:
        st.warning("Pick at least two assets and a period of at least 30 trading days to compare.")
        return

    # 3. Heatmap for the selected period.
    r_table, p_table = correlation_table(period[assets])
    title = f"{start:%b %Y} to {end:%b %Y}"
    st.plotly_chart(make_heatmap(hide_cells(r_table, p_table, cutoff), title))
    st.caption(f"{len(period)} trading days. Blank cells are not significant (p >= {cutoff}).")

    # 4. Fisher Test: The chosen period vs everything that comes before it.
    a, b = "Gold", "Oil"
    before = returns[returns.index < pd.Timestamp(start)]
    if len(before) < 30:
        st.info("Move the start date later to compare against an earlier period.")
        return
    r1, r2 = before[a].corr(before[b]), period[a].corr(period[b])
    z, p = fisher_z_test(r1, len(before), r2, len(period))
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{a}-{b} before {start:%b %Y}", f"r = {r1:+.2f}")
    c2.metric(f"{a}-{b} in this period", f"r = {r2:+.2f}")
    c3.metric("Fisher Z-Test", f"z = {z:.2f}", f"p = {p:.3f}", delta_color="off")

                        