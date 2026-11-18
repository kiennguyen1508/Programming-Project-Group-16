"""
Panel 1: Performance and ratios (Kien).
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from data_loader import get_prices, get_macro, SPLIT

#COLOURS = {"Gold": "#E0B252", "Silver": "#B8C1CF", "Oil": "#4FB39A", "S&P 500": "#9C8CFF"}
COLOURS = {"Gold": "#E0B252", "Silver": "#B8C1CF", "Oil": "#4FB39A", "S&P 500": "#9C8CFF"}
RATIOS = {
    "Gold / Silver" : ("Gold", "Silver"),
    "Gold / Oil" : ("Gold" , "Oil"),
    "Gold / Copper" : ("Gold", "Copper"),
}
TRADING_DAYS = 252 #trading days in a year, used to annualise
RISK_FREE_RATE = 0.04 # assumed yearly cash return, used in the Sharpe ratio and Sortino ratios
VIX_STRESS = 25 # VIX level that counts as a stressed market
REVERSION_WINDOW = 1260 # 5 years of trading days for the rolling mean and σ
MIN_HISTORY = 504 # need at least 2 years of history before testing 

METHOD_NOTES = """
**Data.** Daily closing prices from Yahoo Finance ('yfinance'): gold futures (GC=F),
silver futures (SI=F), Brent crude futures (BZ=F) and the S&P 500 index (^GSPC).
Only days when all four traded are kept.

**Rebasing.** Each price is divided by its first value in the period and multiplied by 100.

**Returns and risk.** Daily simple returns. Annualised return = (1+total return)^(1/years)-1, with years = trading days ÷ 252.
Volatility = standard deviation of daily returns x √252. Max drawdown = the largest fall from a previous peak within the period. Sharpe ratio = (average
daily return x 252 - risk-free rate) ÷ volatility, using the slider rate (default 4%). It uses the average return, not the compound one, because compounding already lowers the return of volatile assets.
Sortino ratio = the same numerator ÷ downside deviation, which only counts days when the return fell below the daily risk-free rate.

**Ratios.** The mean and ±1σ/±2σ bands use the full 2015-2026 history, so they stay fixed when the period changes. The percentile is the share of days since 2015 with a lower ratio than today. The reversion test instead uses a rolling 5-year mean and σ(needing at least 2 years of history), so each day is judged only on data available at the time.

Copper is COMEX copper futures (HG=F, $ per pound) from the shared macro file. The Gold/Copper ratio compares a safe-haven asset with an industrial one: it tends to rise when markets fear a slowdown and fall when they expect growth.

**Calm vs stressed days.** VIX closes from Cboe, via the shared macro file. A day counts as stressed if the VIX closed at or above the chosen level (default 25) on the previous trading day. Yesterday’s VIX is used because today’s VIX rises because stocks are falling today, which would make the comparison circular. Averages from fewer than about 20 days are unreliable.

With few stressed days, one extreme day can dominate the average (e.g. the S&P 500’s +9.5% on 9 April 2025), so compare the average with the median and prefer the Max period for conclusions.

**Limitations.** Futures prices jump slightly when contracts roll over. The S&P 500 is a price index, so it excludes dividends. The ratios are skewed (they can spike far upwards but 
not fall below zero), so the σ bands are only approximate; the percentile is more reliable. Past performance does not predict future returns. Consecutive extreme days overlap, so the real number of independent cases is smaller than "Days" suggests.
"""


@st.cache_data
def load_prices():
    """
    Load the shared price data once, then reuse it on every rerun.
    """
    prices = get_prices()
    return prices.join(get_macro()["Copper"], how="left")

@st.cache_data
def load_vix():
    """
    Daily VIX closes from the shared macro file.
    """
    return get_macro()["VIX"].dropna()

def show_ratios(prices, recent):
    """
    Ratios tabs: ratio chart, z-score and the reversion test.
    """
    name = st.radio("Ratio" , list(RATIOS) , horizontal = True , key = "p1_ratio")
    top, bottom = RATIOS[name]
    ratio = (recent[top] / recent[bottom]).dropna()
    full = (prices[top] / prices[bottom]).dropna()
    stats = ratio_stats(full)
    st.plotly_chart(make_ratio_chart(ratio, name, stats["mean"], stats["std"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Current ratio" , f"{stats['current']:.1f}")
    c2.metric("Z-score" , f"{stats['z']:+.2f}σ",
                help = "How many standard deviations today's ratio is from its 2015-2026 average")
    c3.metric("Percentile" , f"{stats['percentile']:.0f}%",
                help = "Share of days since 2015 when the ratio was lower than today")
    st.info(ratio_summary(name,top,bottom,stats))
    st.markdown("**Do extreme readings revert?**")
    horizon = st.select_slider(
        "Look-ahead horizon",
        options = [21,63,126],
        value = 63,
        format_func=lambda d: f"{d//21} month{'s' if d > 21 else ''}",
        key = "p1_horizon",
    )
    test = reversion_test(full, horizon)
    if test.empty:
        st.caption("The ratio has never been beyond ±2σ of its rolling 5-year man, so there is nothing to test.")
    else:
        st.dataframe(test.style.format({
            "Days" : "{:.0f}",
            "Average change" : "{:+.1%}",
            "Moved back towards mean" : "{:.0%}",
            "Base rate":"{:.0%}",
        }))
        st.caption("Base rate: how often the ratio moved the same way over the same horizon on any day. "
                    "The signal only adds information if 'Moved back towards mean' is clearly higher.")

def show_stress(chosen):
    """
    Stress days tab: average daily return on calm vs stressed days.
    """
    if chosen.empty:
        st.warning("Pick at least one asset to see this.")
        return
    st.markdown("**How did each asset do when markets were scared?**")
    threshold = st.slider("VIX level that counts as stressed", 15, 40,
                    VIX_STRESS, key="p1_vix")
    measure = st.radio("Measure", ["Average", "Median"], horizontal=True, key="p1_vix_measure",
                help="The median ignores a few extreme days; the average includes them.")
    table, n_stressed, n_calm = regime_returns(
        chosen, load_vix(), threshold, "mean" if measure == "Average" else "median")
    st.caption(f"{n_stressed} stressed and {n_calm} calm days in this period. "
               f"A day counts as stressed if the VIX closed at or above {threshold} the day before.")
    if n_stressed == 0:
        st.warning("No stressed days in this period. Try a lower threshold or a longer period.")
    else:
        if min(n_stressed, n_calm) < 20:
            st.warning("One group has fewer than 20 days, so its result is unreliable.")
        st.plotly_chart(make_regime_chart(table, measure))
        st.info(regime_summary(table, measure))

def show_performance(chosen):
    """
    Performance tab: growth chart, drawdown chart, ratio table.
    """
    if chosen.empty:
        st.warning("Pick at least one asset to see this.")
        return
    
    chart_type = st.radio(
        "Chart",
        ["Growth of 100","Drawdown"],
        horizontal = True,
        key = "p1_chart_type",
        label_visibility = "collapsed",
    )
    if chart_type == "Growth of 100":
        log_scale = st.toggle("Log scale", key = "p1_log")
        rebased = chosen / chosen.iloc[0] * 100
        st.plotly_chart(make_price_chart(rebased, log_scale))
    else:
        st.caption("How far each asset is below its previous peak")
        st.plotly_chart(make_drawdown_chart(chosen))

    rf = st.slider(
        "Risk-free rate (%) used in the Sharpe and Sortino ratios",
        min_value=0.0 , max_value=6.0 , value=RISK_FREE_RATE * 100, step=0.25,key="p1_rf", 
    ) / 100
    stats = performance_stats(chosen,rf)
    if len(chosen) < TRADING_DAYS:
        st.caption("This period is shorter than a year, so annualised figures are extrapolated and can look extreme.")
    st.info(price_summary(stats))
    
    higher_is_better = ["Total return", "Annualised return", "Max drawdown",
                        "Sharpe ratio", "Sortino ratio"]
    lower_is_better = ["Volatility (ann.)"]
    best = "font-weight: bold; color:#E5B85C"

    st.dataframe(
        stats.style
        .format({
            "Total return" : "{:.1%}",
             "Annualised return": "{:.1%}",
            "Volatility (ann.)": "{:.1%}",
            "Max drawdown": "{:.1%}",
            "Sharpe ratio": "{:.2f}",
            "Sortino ratio": "{:.2f}",
        })
        .highlight_max(subset=higher_is_better, props=best)
        .highlight_min(subset=lower_is_better, props=best)
    )

def render():
    """
    Draw Panel 1 inside the dashboard.
    """
    st.subheader("Performance & ratios")

    prices = load_prices()

    col_period, col_assets = st.columns([2,3])
    with col_period:
        period = st.radio(
            "Period",
            ["Since 2025" , "YTD" , "1Y" , "5Y" , "Max"],
            horizontal = True,
            key = "p1_period",
        )
    with col_assets:
        selected = st.multiselect(
            "Assets",
            options = list(COLOURS),
            default = list(COLOURS),
            key = "p1_assets",
            help = "Used in the Performance and Stress days tabs. Ratios has its own picker."
        )
    recent = prices.loc[start_date(prices, period):]

    st.caption(
        f"Daily closes from {recent.index[0].date()} to {recent.index[-1].date()} "
        f"({len(recent)} trading days)"
    )
    
    chosen = recent[selected]

    tab_perf, tab_stress, tab_ratio = st.tabs(["Performance","Stress days", "Ratios"])
    with tab_perf:
        show_performance(chosen)
    with tab_stress:
        show_stress(chosen)
    with tab_ratio:
        show_ratios(prices, recent)

    with st.expander("Method and Limitations"):
        st.markdown(METHOD_NOTES)

    st.download_button(
        "Download price data (CSV)",
        data = recent.to_csv().encode("utf-8"),
        file_name= f"panel1_prices_{period.replace(' ', '_').lower()}.csv",
        mime="text/csv",
        key="p1_download"
    )

def overview():
    """
    Small preview of Panel 1 for the overview grid.
    Growth of 100 for all four assets since 2025, chart only
    """
    prices = load_prices()
    recent = prices.loc[start_date(prices, "Since 2025"):, list(COLOURS)]
    rebased = recent / recent.iloc[0] * 100

    # Same chart as the full panel, made smaller and without the tool bar
    fig = make_price_chart(rebased)
    fig.update_layout(height=220,yaxis_title=None,
                      margin=dict(l=0,r=0,t=30,b=0))
    fig.update_yaxes(nticks=3)
    st.plotly_chart(fig, config={"staticPlot" : True},
                    key="p1_overview_chart")
    

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

def make_drawdown_chart(prices):
    """
    How far each asset is below its previous peak on each day (0% = at a new high).
    """
    drawdown = (prices / prices.cummax() - 1) * 100
    fig = go.Figure()
    for asset in drawdown.columns:
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown[asset],
            name=asset,
            mode="lines",
            line=dict(color=COLOURS[asset],width=1.5),
            hovertemplate="%{y:.1f}%",
        ))
    fig.update_layout(
        height=260,
        hovermode="x unified",
        yaxis_title="Below previous peak (%)",
        legend = dict(orientation = "h" , y = 1.1),
        margin=dict(l=10,r=10,t=30,b=10),
    )
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

def reversion_test(full, horizon_days):
    """
    What happened to the ratio after it was more than 2σ from its rolling 5-year mean?
    Each day uses only data available at the time, so there is no look-ahead bias
    """
    rolling = full.rolling(REVERSION_WINDOW, min_periods=MIN_HISTORY)
    mean, std = rolling.mean(), rolling.std()
    future = full.shift(-horizon_days) / full - 1 # % change over the next N days
    usable = mean.notna() & future.notna()
    rows = {}
    for label, mask, back in [
        ("Above +2σ" , full > mean + 2 * std, future < 0),
        ("Below -2σ" , full < mean -2 * std , future >0),
    ]:
        moves = future[mask & usable]
        if moves.empty:
            continue
        rows[label] = {
            "Days" : len(moves),
            "Average change" : moves.mean(),
            "Moved back towards mean" : back[moves.index].mean(),
            "Base rate": back[usable].mean()
        }
    return pd.DataFrame(rows).T


def performance_stats(prices, risk_free = RISK_FREE_RATE):
    """
    Return, risk and risk-adjusted return for each asset over the period.
    """
    returns = prices.pct_change().dropna()
    years = len(returns) / TRADING_DAYS
    total = prices.iloc[-1] / prices.iloc[0] - 1
    annual = (1 + total) ** (1 / years) -1          # compound growth, shown in the table
    mean_annual = returns.mean() * TRADING_DAYS     # average return, used in Sharpe/Sortino
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    daily_rf = risk_free / TRADING_DAYS
    shortfall = (returns - daily_rf).clip(upper=0) #only days below the risk-free rate
    downside_vol = np.sqrt((shortfall**2).mean()) * np.sqrt(TRADING_DAYS)
    drawdown = (prices / prices.cummax() - 1).min()
    return pd.DataFrame({
        "Total return" : total,
        "Annualised return" : annual,
        "Volatility (ann.)" : vol,
        "Max drawdown" : drawdown,
        "Sharpe ratio" : (mean_annual - risk_free) / vol,
        "Sortino ratio" : (mean_annual - risk_free) / downside_vol,
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


def regime_returns(prices, vix, threshold, measure="mean"):
    """
    Average daily return of each asset on calm vs stressed days.
    A day counts as stressed if the VIX closed at or above the threshold the day before.
    """
    returns = prices.pct_change().dropna()
    prev_vix = vix.shift(1).reindex(returns.index)
    valid = prev_vix.notna()
    returns, prev_vix = returns[valid], prev_vix[valid]
    stressed = prev_vix >= threshold
    table = pd.DataFrame({
        "Calm days": returns[~stressed].agg(measure),
        "Stressed days": returns[stressed].agg(measure),
    })
    return table, int(stressed.sum()),int((~stressed).sum())

def make_regime_chart(table, measure):
    """
    Grouped bars: each asset's average daily return on calm vs stressed days.
    """
    fig = go.Figure()
    for col, colour in [("Calm days","#94A3B8"),("Stressed days","#DC2626")]:
        fig.add_trace(go.Bar(
            x=table.index, y=table[col] * 100, name=col,
            marker_color=colour, hovertemplate="%{y:.2f}%",
        ))
    fig.add_hline(y=0, line_color="grey")
    fig.update_layout(
        barmode="group", height=320,
        yaxis_title=f"{measure} daily return (%)",
        legend = dict(orientation="h", y=1.1),
        margin=dict(l=10,r=10,t=30,b=10),
    )
    return fig

def regime_summary(table, measure):
    """
    One-sentence takeaway from the calm vs stressed comparison.
    """
    best = table["Stressed days"].idxmax()
    worst = table["Stressed days"].idxmin()
    return (f"On stressed days, **{best}** had the highest {measure.lower()} return "
        f"({table.loc[best, 'Stressed days']:+.2%} a day) and **{worst}** the lowest "
        f"({table.loc[worst, 'Stressed days']:+.2%}).")
    
