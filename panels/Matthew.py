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




NAMELAYOUT = {"S&P 500": "S&P<br>500", "Real Yield": "Real<br>Yield",
"Commodities": "Commod-<br>ities"}




def make_heatmap(r_table: pd.DataFrame, title: str):
    """Correlation heatmap with each asset's name written on the diagonal."""
    fig = px.imshow(r_table.round(2), text_auto=True, color_continuous_scale="RdBu_r", 
                    zmin=-1, zmax=1, title=title, aspect="auto")                 # change square to rectangle
    for name in r_table.index:
        fig.add_annotation(x=name, y=name, text=f"<b>{NAMELAYOUT.get(name, name)}</b>", 
                           showarrow=False, font=dict(size=12))
        fig.update_xaxes(showticklabels=False, showgrid=False, ticks= "")
        fig.update_yaxes(showticklabels=False, showgrid=False, ticks= "")
        fig.update_traces(textfont_size=13)
        fig.update_layout(height=650, margin=dict(l=10, r=10, t=50, b=10),
                          coloraxis_colorbar=dict(thickness=12, title="r"))
        return fig
    

def hide_cells(r_table: pd.DataFrame, p_table: pd.DataFrame, cutoff: float = 0.05):
    """Blank out the repeated half, and other non statisitically significant cells."""
    bottom_half = np.tril(np.ones(r_table.shape, dtype=bool), k=-1)
    diagonal = np.eye(len(r_table), dtype=bool)
    not_significant = p_table.values >= cutoff
    return r_table.mask(bottom_half | diagonal | not_significant)
    

def fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> tuple:
    """Test whether two correlations from separate periods are really different."""
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1 / (n1-3) + 1 / (n2 - 3))
    z =(z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

def make_rolling_chart(returns: pd.DataFrame, a: str, b: str, window: int, start):
    """Line chart of the correlation between a and b, calculated over a sliding window period"""
    rolling = returns[a].rolling(window).corr(returns[b]).dropna()
    fig = go.Figure(go.Scatter(x=rolling.index, y=rolling, mode="lines", name=f"{a}-{b}"))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.add_vline(x=pd.Timestamp(start), line_dash="dot", line_color="orange")
    fig.update_layout(title=f"{a}-{b}: {window}-day rolling correlation",
                      yaxis_title="Correlation (r)", yaxis_range=[-1,1])
    return fig

def make_animated_heatmap(returns: pd.DataFrame, assets: list, cutoff: float):
    """One heatmap per calendar year, plays like a slidehow."""
    years = sorted(returns.index.year.unique())
    frames = []
    for year in years:
        r_table, p_table = correlation_table(returns.loc[str(year), assets])
        frames.append(hide_cells(r_table,p_table, cutoff).round(2))
    fig = px.imshow(np.stack(frames), animation_frame=0, text_auto=True,
                    x=list(frames[0].columns),
    y=list(frames[0].index),
    color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    for frame, step, year in zip(fig.frames, fig.layout.sliders[0].steps, years):
        frame.name = str(year)
        step.args = [[str(year)], step.args[1]]
        step.label = str(year)
    fig.layout.sliders[0].currentvalue.prefix = "Year: "
    play = fig.layout.updatemenus[0].buttons[0].args[1]
    play["frame"]["duration"] = 1500
    play["transition"]["duration"] = 500
    for name in assets:
        fig.add_annotation(x=name, y=name, text=f"<b>{NAMELAYOUT.get(name, name)}</b>",
                         showarrow=False, font=dict(size=12))
    fig.update_xaxes(showticklabels=False, showgrid=False, ticks="")
    fig.update_yaxes(showticklabels=False, showgrid=False, ticks="")
    fig.update_traces(textfont_size=13)
    fig.update_layout(height=550, margin=dict(l=10, r=10, t=50, b=10),
                      coloraxis_colorbar=dict(thickness=12, title="r"))
    return fig

def result_sentence(a, b, r1, r2, p, cutoff, start) -> str:
    """Write a plain-english summary of the Fisher Test result."""
    direction = "flipped" if np.sign(r1) != np.sign(r2) else "moved"
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    if p < cutoff:
        verdict = f"this change is statistically significant ({p_text})"
    else:
        verdict = f"this change could be chance ({p_text}), so there is no clear sign they decoupled"
        return (f"**Result:** The {a}-{b} correlation {direction} from {r1:+.2f} before ")
        f"{start:%b %Y} to {r2:+.2f} in this period; {verdict}."

def render():
    """Draw Panel 3 on the dashboard"""
    st.subheader("Correlation Heatmap")
    returns = get_returns().join(load_macro(), how="inner").dropna()

    # 1. Controls
    view = st.radio("View", ["Heatmap", "Rolling", "Animated"], horizontal=True, key="p3_view")
    assets = st.multiselect("Assets", list(returns.columns),
                        default=["Gold", "Silver", "Oil", "S&P 500", "Real yield"],
                        key="p3_assets")
    first, last = returns.index.min().date(), returns.index.max().date()
    start, end = st.slider("Time Period", min_value=first, max_value=last,
                           value=(pd.Timestamp(SPLIT).date(), last),
                           format="MMM YYYY", key="p3_dates")
    cutoff = st.slider("Significance level (p-value cutoff)", min_value=0.01,
                       max_value=0.20, value=0.05, step=0.01, format="%.2f", key="p3_cutoff")
    pick1, pick2 = st.columns(2)
    a = pick1.selectbox("Compare", assets, index=0, key="p3_pair_a")
    b = pick2.selectbox("with", assets, index=min(2, len(assets) - 1), key="p3_pair_b")

                        

    # 2. Checking the choices make sense.
    period = returns.loc[str(start):str(end)]
    if len(assets) < 2 or len(period) < 30:
        st.warning("Please pick at least two assets and a period of at least 30 trading days to compare.")
        return

    # 3. Heatmap for the selected period.
    if view == "Heatmap":
        r_table, p_table = correlation_table(period[assets])
        title = f"{start:%b %Y} to {end:%b %Y}"
        st.plotly_chart(make_heatmap(hide_cells(r_table, p_table, cutoff), title), width="stretch")
        st.caption(f"{len(period)} trading days. Blank cells are not significant (p >= {cutoff}).")
    elif view == "Animated":
        st.plotly_chart(make_animated_heatmap(returns, assets, cutoff), width="stretch")
        st.caption("Press play to step through each calendar year. This view uses every year, "
                   "so the time period slider doesn't apply. The year 2026 has only partial data.")
    else:
        window = st.slider("Rolling Window (trading days)", min_value=30, max_value=250,
                           value=90, step=10, key="p3_window")
        st.plotly_chart(make_rolling_chart(returns, a, b, window, start))
        st.caption(f"Each point is the {a}-{b} correlation over the previous {window} trading days."
                   f"The dotted line marks the start of your chosen period.")

    # 4. Fisher Test: The chosen period vs everything that comes before it.
    if a == b:
        st.info("Please select two different assets to run the Fisher Test.")
        return
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
    st.info(result_sentence(a, b, r1, r2, p, cutoff, start))

    # 5. Method and Limitations Dropdown box (if needed?, not sure yet but as a test will add.)
    with st.expander("Method and Limitations"):
        st.markdown("""
This panel compares the daily returns of gold, silver, oil, the S&P 500
and other key macro measure since 2015. Each cell shows the Pearson correlation
for the chosen period, and cells that could be down to chance (with p-values
above the cutoff that you select from 0.01 to 0.20 (for now)) are hidden.
Correlation is not causation, and short periods give less reliable results etc etc.
This is just a test paragraph for now, I'm not sure if this is necessary. 
More detail can be added later if we decide.
""")


def overview():
    """Small preview for our opening page, shows all the assets for the whole period."""
    returns = get_returns().join(load_macro(), how="inner").dropna()
    r_table, p_table = correlation_table(returns)
    fig = make_heatmap(hide_cells(r_table, p_table), None)
    fig.update_layout(height=370, margin=dict(l=0, r=0, t=10, b=0))
    fig.update_traces(textfont_size=11)
    fig.update_annotations(font_size=10)
    st.plotly_chart(fig, width="stretch", key="p3_preview", config={"displayModeBar": False})