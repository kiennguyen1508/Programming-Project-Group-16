"""
Panel 1: Performance and ratios (Kien).
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from data_loader import get_prices, SPLIT


COLOURS = {"Gold": "#B45309", "Silver": "#7C8798", "Oil": "#0F766E", "S&P 500": "#2563EB"}
RATIOS = {
    "Gold / Silver" : ("Gold", "Silver"),
    "Gold / Oil" : ("Gold" , "Oil"),
}
TRADING_DAYS = 252 #trading days in a year, used to annualise
RISK_FREE_RATE = 0.04 # assumed early cash return, used in the Sharpe ratio

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
        stats = performance_stats(chosen)
        st.info(price_summary(stats))
        st.dataframe(
            stats.style.format({
                "Total return": "{:.1%}",
                "Annualised return": "{:.1%}",
                "Volatility (ann.)": "{:.1%}",
                "Max drawdown": "{:.1%}",
                "Sharpe ratio": "{:.2f}",
            })
        )

    else:
        name = st.radio("Ratio" , list(RATIOS) , horizontal = True , key = "p1_ratio")
        top, bottom = RATIOS[name]
        ratio = recent[top] / recent[bottom]
        full = prices[top] / prices[bottom]
        stats = ratio_stats(full)
        st.plotly_chart(make_ratio_chart(ratio, name, stats["mean"], stats["std"]))

        c1, c2, c3 = st.columns(3)
        c1.metric("Current ratio" , f"{stats['current']:.1f}")
        c2.metric("Z-score" , f"{stats['z']:+.2f}σ",
                  help = "How many standard deviations today's ratio is from its 2015-2026 average")
        c3.metric("Percentile" , f"{stats['percentile']:.0f}%",
                  help = "Share of days since 2015 when the ratio was lower than today")
        st.info(ratio_summary(name,top,bottom,stats))

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


def ratio_stats(full):
    """
    Where today's ratio sits compared with its full history since 2015.
    """
    current = full.iloc[-1]
    mean, std = full.mean(), full.std()
    return{
        "current" : current,
        "mean" : mean,
        "std" : std,
        "z" : (current - mean) / std,
        "percentile" : (full < current).mean() * 100,
    }

def performance_stats(prices):
    """
    Return, risk and risk-adjusted return for each asset over the period.
    """
    returns = np.log(prices).diff().dropna()
    years = len(returns) / TRADING_DAYS
    total = prices.iloc[-1] / prices.iloc[0] - 1
    annual = (1 + total) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    drawdown = (prices / prices.cummax() -1).min()
    return pd.DataFrame({
        "Total return" : total,
        "Annualised return" : annual,
        "Volatility (ann.)": vol,
        "Max drawdown" : drawdown,
        "Sharpe ratio" : (annual - RISK_FREE_RATE) / vol,
    }).rename_axis("Asset")


def price_summary(stats):
    """One-sentence takeaway from the performance table."""
    if stats.empty:
        return ""
    best = stats["Total return"].idxmax()
    safest = stats["Sharpe ratio"].idxmax()
    deepest = stats["Max drawdown"].idxmin()
    if best == safest:
        first = (f"**{best}** returned the most ({stats.loc[best, 'Total return']:+.0%}) "
                 f"and also had the best risk-adjusted return "
                 f"(Sharpe {stats.loc[best, 'Sharpe ratio']:.2f}).")
    else:
        first = (f"**{best}** returned the most ({stats.loc[best, 'Total return']:+.0%}), "
                 f"but **{safest}** had the best risk-adjusted return "
                 f"(Sharpe {stats.loc[safest, 'Sharpe ratio']:.2f}).")
    second = (f" **{deepest}** had the deepest fall from a peak "
              f"({stats.loc[deepest, 'Max drawdown']:.0%}).")
    return first + second


def ratio_summary(name, top, bottom, stats):
    """One-sentence reading of where the ratio sits today."""
    z = stats["z"]
    if z > 2:
        level = "at a historical extreme high"
    elif z > 1:
        level = "above its long-run average"
    elif z < -2:
        level = "at a historical extreme low"
    elif z < -1:
        level = "below its long-run average"
    else:
        level = "close to its long-run average"
    return (f"The {name} ratio is **{stats['current']:.1f}**, {level}: higher than on "
            f"{stats['percentile']:.0f}% of days since 2015. A high ratio means "
            f"{bottom.lower()} is cheap relative to {top.lower()}.")

