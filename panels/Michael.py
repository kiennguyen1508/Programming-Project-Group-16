"""
Panel 4 — Distribution & Risk (Michael)

Main visual: interactive 3D VOLCANO / joint probability-density surface.
X = daily return (%)
Y = trading-volume percentile (0–100, calculated within each asset)
Z = joint probability density of return and trading activity.

Volume is fetched from Yahoo Finance. Where the index/futures contract has no
usable volume history, a clearly labelled liquid ETF volume proxy is used.
Volume percentiles, rather than raw share/contract counts, make activity
positions comparable without pretending the underlying markets have equal size.

This panel compares asset-period distributions and links them to downside-risk
statistics, tail tests, historical VaR breaches and rolling volatility.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import gaussian_kde

from data_loader import get_returns, get_macro, get_macro_changes

try:
    import yfinance as yf
except ImportError:
    yf = None

SPLIT_DATE = pd.Timestamp("2025-01-01")
YEAR_2026_START = pd.Timestamp("2026-01-01")
TRADING_DAYS = 252

ASSET_ORDER = ["Gold", "Silver", "Copper", "Oil", "S&P 500"]
ASSET_COLOURS = {
    "Gold": "#E0B252",
    "Silver": "#B8C1CF",
    "Copper": "#C47B4B",
    "Oil": "#4FB39A",
    "S&P 500": "#9C8CFF",
}
PERIODS = ["2015–2024", "2025", "2026 YTD", "Since 2025"]
PERIOD_COLOURS = {
    "2015–2024": "#718096",
    "2025": "#E0A82E",
    "2026 YTD": "#42A5F5",
    "Since 2025": "#A78BFA",
}
# Primary Yahoo symbols are the same market series used by the group where possible.
# ETF alternatives are explicitly labelled proxies for trading activity only.
VOLUME_SYMBOLS = {
    "Gold": [("GC=F", "Gold futures volume"), ("GLD", "GLD ETF volume proxy")],
    "Silver": [("SI=F", "Silver futures volume"), ("SLV", "SLV ETF volume proxy")],
    "Copper": [("HG=F", "COMEX copper futures volume"), ("CPER", "CPER ETF volume proxy")],
    "Oil": [("BZ=F", "Brent futures volume"), ("USO", "USO ETF volume proxy")],
    "S&P 500": [("SPY", "SPY ETF volume proxy")],
}

METHOD_NOTES = """
**3D volcano.** The surface is a two-dimensional Gaussian kernel density
estimate (KDE) of daily return and trading activity. X is daily log return in
percent; Y is the asset's trading-volume percentile over its available history;
Z is joint probability density. The volcano peak marks combinations of return
and activity that occurred most frequently. It is a joint distribution, not a
forecast or a causal relationship.

**Volume treatment.** Futures volume is used where Yahoo Finance provides
enough observations. Otherwise the panel uses a named ETF proxy (GLD, SLV,
CPER, USO or SPY). Volume is converted to a percentile within each instrument
so that 90 means unusually high activity for that instrument, not the same
number of contracts or shares. ETF proxy volume is a measure of related-market
activity and is not the exact volume of the futures/index series whose returns
are plotted.

**Risk.** Annualised volatility = daily standard deviation × √252. 95% VaR is
the fifth-percentile loss threshold; Expected Shortfall is the average loss
among returns at or below that threshold. Skewness measures asymmetry and
excess kurtosis measures tail heaviness. Maximum drawdown is calculated from
the cumulative wealth path reconstructed from log returns.

**Tests.** KS compares the pre-2025 and post-2024 return distributions;
Levene compares their variances. Standard p-values assume independent
observations, while financial returns can exhibit volatility clustering.
Interpret p-values alongside sample sizes, effect sizes and the graph.

**Limitations.** 2026 is a partial year. Different markets have different
trading hours and holidays. Futures can be affected by contract rolls. KDE
shape depends on bandwidth and sample size. S&P 500 index returns exclude
dividends. These historical patterns do not predict future returns.
"""


@st.cache_data(show_spinner=False, max_entries=10)
def load_shared_data():
    returns = get_returns().copy()
    macro = get_macro().copy()
    macro_changes = get_macro_changes().copy()

    for frame in (returns, macro, macro_changes):
        frame.index = pd.to_datetime(frame.index)
        if frame.index.tz is not None:
            frame.index = frame.index.tz_localize(None)
        frame.sort_index(inplace=True)

    if "Copper" not in returns.columns:
        if "Copper" in macro_changes.columns:
            returns = returns.join(macro_changes["Copper"].rename("Copper"), how="left")
        elif "Copper" in macro.columns:
            returns = returns.join(macro["Copper"].pct_change().rename("Copper"), how="left")

    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns


@st.cache_data(ttl=86400, show_spinner=False, max_entries=500)
def download_one_volume_series(symbol, start_date, end_date):
    """Download daily volume from Yahoo Finance; return an empty series on failure."""
    if yf is None:
        return pd.Series(dtype=float, name=symbol)
    try:
        hist = yf.Ticker(symbol).history(
            start=start_date, end=end_date, auto_adjust=False, actions=False
        )
        if hist.empty or "Volume" not in hist.columns:
            return pd.Series(dtype=float, name=symbol)
        volume = pd.to_numeric(hist["Volume"], errors="coerce")
        idx = pd.to_datetime(hist.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        volume.index = idx.normalize()
        volume = volume[volume > 0].groupby(level=0).last()
        volume.name = symbol
        return volume
    except Exception:
        return pd.Series(dtype=float, name=symbol)


@st.cache_data(ttl=86400, show_spinner=False, max_entries=20)
def load_volume_data(start_date, end_date):
    """Use futures volume where available; otherwise a labelled ETF proxy."""
    series_by_asset = {}
    source_by_asset = {}
    for asset, candidates in VOLUME_SYMBOLS.items():
        chosen = pd.Series(dtype=float)
        chosen_source = "Unavailable"
        for symbol, label in candidates:
            candidate = download_one_volume_series(symbol, start_date, end_date)
            if len(candidate) >= 60:
                # Futures data can be sparse on Yahoo; require a reasonable coverage.
                chosen = candidate
                chosen_source = label
                if symbol.endswith("=F") and candidate.notna().sum() >= 250:
                    break
                if symbol in ("SPY", "GLD", "SLV", "CPER", "USO"):
                    break
        if not chosen.empty:
            # Rank against the full downloaded history, not separately by year.
            percentile = chosen.rank(pct=True) * 100
            percentile.name = asset
            series_by_asset[asset] = percentile
        source_by_asset[asset] = chosen_source
    if series_by_asset:
        return pd.DataFrame(series_by_asset), source_by_asset
    return pd.DataFrame(), source_by_asset


def clean_series(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def period_slice(data, period):
    if period == "2015–2024":
        return data.loc[data.index < SPLIT_DATE]
    if period == "2025":
        return data.loc[(data.index >= SPLIT_DATE) & (data.index < YEAR_2026_START)]
    if period == "2026 YTD":
        return data.loc[data.index >= YEAR_2026_START]
    if period == "Since 2025":
        return data.loc[data.index >= SPLIT_DATE]
    return data.iloc[0:0]

def make_volcano_3d(returns, volumes, distribution_specs, bandwidth=1.0):
    """
    Draw one or two genuine 3D joint-distribution surfaces.
    Each surface is a 2D KDE: X=daily return, Y=relative trading volume,
    Z=joint probability density. Comparisons are side-by-side, never stacked
    on top of one another, so the shape of each distribution remains visible.
    """
    pairs = []
    for asset, period in distribution_specs:
        if asset not in returns.columns or asset not in volumes.columns:
            continue
        r = period_slice(returns[[asset]], period)[asset]
        v = volumes[asset].reindex(r.index)
        data = pd.concat(
            [r.rename("return"), v.rename("volume_percentile")], axis=1
        ).dropna()
        if len(data) >= 50 and data["return"].std() > 0 and data["volume_percentile"].std() > 0:
            pairs.append((asset, period, data))

    if not pairs:
        return None, (
            "Not enough aligned daily-return and trading-volume observations. "
            "Try another asset or a longer period."
        )

    # Use the same X and Y grid for every selected distribution.
    pooled_returns = np.concatenate(
        [item[2]["return"].to_numpy(dtype=float) * 100 for item in pairs]
    )
    x_lo, x_hi = np.nanpercentile(pooled_returns, [0.5, 99.5])
    x_pad = max((x_hi - x_lo) * 0.15, 0.2)
    x_grid = np.linspace(x_lo - x_pad, x_hi + x_pad, 90)
    y_grid = np.linspace(0, 100, 70)
    xx, yy = np.meshgrid(x_grid, y_grid)

    fig = make_subplots(
        rows=1,
        cols=len(pairs),
        specs=[[{"type": "scene"} for _ in pairs]],
        subplot_titles=[f"{asset} · {period}" for asset, period, _ in pairs],
        horizontal_spacing=0.04,
    )

    surfaces_added = 0
    for col, (asset, period, data) in enumerate(pairs, start=1):
        observations = data[["return", "volume_percentile"]].to_numpy(dtype=float)
        observations[:, 0] *= 100

        # Deterministic cap keeps the interactive surface responsive.
        if len(observations) > 6000:
            rng = np.random.default_rng(17)
            observations = observations[
                rng.choice(len(observations), 6000, replace=False)
            ]

        try:
            kde = gaussian_kde(
                observations.T,
                bw_method=lambda fitted: fitted.scotts_factor() * bandwidth,
            )
            positions = np.vstack([xx.ravel(), yy.ravel()])
            zz = kde(positions).reshape(xx.shape)
        except (np.linalg.LinAlgError, ValueError):
            continue

        surfaces_added += 1
        label = f"{asset} · {period}"
        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=zz,
                name=label,
                colorscale="Turbo",
                showscale=False,
                opacity=1.0,
                lighting=dict(
                    ambient=0.52,
                    diffuse=0.88,
                    specular=0.45,
                    roughness=0.42,
                ),
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=False,
                        color="rgba(255,255,255,0.48)",
                        width=1,
                    )
                ),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Daily return: %{x:.2f}%<br>"
                    "Trading-volume percentile: %{y:.0f}<br>"
                    "Joint probability density: %{z:.5f}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )

    if surfaces_added == 0:
        return None, "The joint distributions could not be estimated from the available observations."

    fig.update_layout(
        title=dict(
            text="Joint probability distribution: daily returns × trading activity",
            x=0.02,
        ),
        height=720 if len(pairs) == 1 else 760,
        margin=dict(l=0, r=0, t=90, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE2EC"),
    )
    fig.update_scenes(
        xaxis=dict(
            title="Daily price change / return (%)",
            range=[x_lo - x_pad, x_hi + x_pad],
            gridcolor="rgba(150,160,180,0.22)",
            zeroline=True,
        ),
        yaxis=dict(
            title="Trading volume (within-asset percentile)",
            range=[0, 100],
            gridcolor="rgba(150,160,180,0.22)",
        ),
        zaxis=dict(
            title="Probability density",
            rangemode="tozero",
            gridcolor="rgba(150,160,180,0.22)",
        ),
        aspectmode="manual",
        aspectratio=dict(x=1.35, y=1.0, z=0.72),
        camera=dict(eye=dict(x=1.55, y=1.65, z=1.0)),
        bgcolor="rgba(0,0,0,0)",
    )
    return fig, None

def max_drawdown_from_log_returns(log_returns):
    r = clean_series(log_returns)
    if r.empty:
        return np.nan
    wealth = np.exp(r.cumsum())
    return float((wealth / wealth.cummax() - 1).min())


def risk_statistics(series):
    r = clean_series(series)
    if len(r) < 10:
        return None
    q05 = float(r.quantile(0.05))
    tail = r[r <= q05]
    return {
        "Observations": int(len(r)),
        "Mean daily return (%)": float(r.mean() * 100),
        "Annualised volatility (%)": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100),
        "95% VaR (%)": float(-q05 * 100),
        "95% Expected Shortfall (%)": float(-tail.mean() * 100),
        "Skewness": float(stats.skew(r, bias=False)),
        "Excess kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)),
        "Maximum drawdown (%)": float(max_drawdown_from_log_returns(r) * 100),
    }


def risk_table(returns, assets, periods):
    rows = []
    for period in periods:
        subset = period_slice(returns, period)
        for asset in assets:
            if asset in subset:
                result = risk_statistics(subset[asset])
                if result:
                    rows.append({"Asset": asset, "Period": period, **result})
    return pd.DataFrame(rows)


def make_rolling_volatility(returns, assets, window=30):
    fig = go.Figure()
    for asset in assets:
        rolling = (
            returns[asset]
            .rolling(window, min_periods=max(15, window // 2))
            .std()
            * np.sqrt(TRADING_DAYS)
            * 100
        )
        fig.add_trace(
            go.Scatter(
                x=rolling.index,
                y=rolling,
                mode="lines",
                name=asset,
                line=dict(
                    color=ASSET_COLOURS.get(asset, "#A0AAB8"),
                    width=2,
                ),
            )
        )
    fig.add_vline(
        x=SPLIT_DATE,
        line_dash="dash",
        line_color="#E0B252",
        annotation_text="Start of 2025",
    )
    fig.update_layout(
        title=f"{window}-day rolling annualised volatility",
        height=380,
        xaxis_title=None,
        yaxis_title="Annualised volatility (%)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def tests_table(returns, assets):
    before = period_slice(returns, "2015–2024")
    after = period_slice(returns, "Since 2025")
    rows = []
    for asset in assets:
        x, y = clean_series(before[asset]), clean_series(after[asset])
        if len(x) >= 30 and len(y) >= 20:
            ks = stats.ks_2samp(x, y)
            lev = stats.levene(x, y, center="median")
            rows.append({
                "Asset": asset, "KS statistic": ks.statistic, "KS p-value": ks.pvalue,
                "Levene statistic": lev.statistic, "Levene p-value": lev.pvalue,
                "Pre-2025 n": len(x), "Post-2024 n": len(y)
            })
    return pd.DataFrame(rows)


def var_breach_table(returns, assets):
    before = period_slice(returns, "2015–2024")
    after = period_slice(returns, "Since 2025")
    rows = []
    for asset in assets:
        base, later = clean_series(before[asset]), clean_series(after[asset])
        if len(base) >= 100 and len(later) >= 20:
            threshold = base.quantile(0.05)
            breach = later < threshold
            rows.append({
                "Asset": asset, "Historical 95% VaR loss (%)": -threshold * 100,
                "Post-2024 breach rate (%)": breach.mean() * 100,
                "Expected breach rate (%)": 5.0, "Breach days": int(breach.sum()),
                "Post-2024 observations": len(later)
            })
    return pd.DataFrame(rows)


def render():
    st.subheader("Distribution & Risk")
    st.caption(
        "Research question: Did the relationship between daily price changes "
        "and trading activity change after 2025, and was the change unique to Gold?"
    )

    try:
        returns = load_shared_data()
    except Exception as exc:
        st.error(f"Could not load shared project data: {exc}")
        return

    assets = [a for a in ASSET_ORDER if a in returns.columns]
    if "Gold" not in assets:
        st.error(f"Gold is missing from get_returns(). Available columns: {list(returns.columns)}")
        return

    first_date = returns.index.min().date()
    last_date = (returns.index.max() + pd.Timedelta(days=1)).date()
    with st.spinner("Loading trading-volume histories from Yahoo Finance..."):
        volumes, volume_sources = load_volume_data(str(first_date), str(last_date))

    st.markdown("### 1. Interactive 3D probability distribution")
    st.write(
        "This is a joint distribution of daily price changes and trading activity. "
        "The surface rises where that combination occurred most frequently. "
        "Choose one distribution first; optionally compare it with a second "
        "distribution in a separate 3D scene."
    )

    c1, c2, c3 = st.columns([1.25, 1.15, 1])
    with c1:
        selected_asset = st.selectbox(
            "Asset",
            assets,
            index=assets.index("Gold"),
            key="p4_primary_asset",
        )
    with c2:
        selected_period = st.segmented_control(
            "Year / period",
            PERIODS[:3],
            default=PERIODS[0],
            key="p4_primary_period",
        )
    with c3:
        bandwidth = st.select_slider(
            "Surface smoothness",
            options=[0.65, 0.8, 1.0, 1.2, 1.4],
            value=1.0,
            format_func=lambda x: f"{x:.2f}×",
            key="p4_volcano_bandwidth",
        )

    compare_enabled = st.checkbox(
        "Compare with a second asset or year",
        value=False,
        key="p4_compare_enabled",
    )
    distribution_specs = [(selected_asset, selected_period)]

    if compare_enabled:
        cc1, cc2 = st.columns(2)
        with cc1:
            compare_asset = st.selectbox(
                "Comparison asset",
                assets,
                index=assets.index("Silver") if "Silver" in assets else 0,
                key="p4_compare_asset",
            )
        with cc2:
            compare_period = st.segmented_control(
                "Comparison year / period",
                PERIODS[:3],
                default=PERIODS[1],
                key="p4_compare_period",
            )
        distribution_specs.append((compare_asset, compare_period))

    chosen_assets = list(dict.fromkeys(asset for asset, _ in distribution_specs))
    chosen_periods = list(dict.fromkeys(period for _, period in distribution_specs))

    if not volumes.empty:
        source_lines = [
            f"**{a}:** {volume_sources.get(a, 'Unavailable')}"
            for a in chosen_assets
        ]
        st.caption("Volume sources — " + " | ".join(source_lines))

    missing_volume = [a for a in chosen_assets if a not in volumes.columns]
    if missing_volume:
        st.warning(
            "Volume unavailable for: " + ", ".join(missing_volume)
            + ". Check the Yahoo Finance symbols or remove the unavailable asset."
        )
    elif chosen_assets:
        fig, error = make_volcano_3d(
            returns, volumes, distribution_specs, bandwidth
        )
        if error:
            st.warning(error)
        else:
            st.plotly_chart(
                fig,
                width="stretch",
                key="p4_volcano_3d",
                config={"scrollZoom": True, "displaylogo": False},
            )

    st.caption(
        "X = daily percentage return; Y = trading-volume percentile within that "
        "instrument (0–100); Z = estimated joint probability density. The "
        "percentile normalises activity within each market, so it compares "
        "unusually quiet versus unusually busy days rather than raw contract "
        "or share counts. When comparison is enabled, the distributions are "
        "shown side-by-side rather than obscuring one another."
    )

    with st.expander("How to interpret the 3D distribution", expanded=True):
        st.markdown(
            "- **High peak:** that combination of daily return and trading activity occurred frequently.\n"
            "- **Wide spread along X:** daily returns are more dispersed, including larger gains or losses.\n"
            "- **Wide spread along Y:** returns occur across a broader range of trading activity.\n"
            "- **Long extension towards negative X:** larger downside movements occur in the distribution.\n"
            "- **Separate peaks or ridges:** more than one common return/activity regime may be present.\n"
            "- Compare the same asset before and after 2025, then compare whether peer assets changed by a similar amount."
        )

    st.markdown("### 2. Risk measures — quantify what the volcano shows")
    selected_for_risk = chosen_assets or assets
    periods_for_risk = chosen_periods or ["2015–2024", "2025", "2026 YTD"]
    table = risk_table(returns, selected_for_risk, periods_for_risk)
    if not table.empty:
        numeric = [c for c in table.columns if c not in ["Asset", "Period", "Observations"]]
        table[numeric] = table[numeric].round(3)
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.info("Not enough observations for the selected risk statistics.")
    st.caption(
        "VaR and Expected Shortfall are daily losses; larger values mean greater "
        "downside risk. Volatility is annualised. Kurtosis is excess kurtosis."
    )

    st.markdown("### 3. Did historical downside-risk limits still hold?")
    breaches = var_breach_table(returns, selected_for_risk)
    if not breaches.empty:
        st.dataframe(breaches.round(3), hide_index=True, width="stretch")
    else:
        st.info("Insufficient pre-2025 or post-2024 history to backtest 95% VaR.")
    st.caption(
        "Each asset's 2015–2024 5th-percentile loss threshold is held fixed and "
        "tested against 2025 onwards. A breach rate above 5% means the historical "
        "95% VaR threshold was exceeded more often than its nominal rate."
    )

    st.markdown("### 4. When did risk change?")
    window = st.select_slider(
        "Rolling volatility window", options=[21, 30, 60, 90],
        value=30, format_func=lambda d: f"{d} trading days",
        key="p4_volcano_rolling_window"
    )
    st.plotly_chart(
        make_rolling_volatility(returns, selected_for_risk, window),
        width="stretch", key="p4_volcano_rolling"
    )

    st.markdown("### 5. Are the distribution changes statistically detectable?")
    tests = tests_table(returns, selected_for_risk)
    if not tests.empty:
        st.dataframe(tests.round(4), hide_index=True, width="stretch")
    else:
        st.info("Insufficient observations for the pre-2025 versus post-2024 tests.")
    st.caption(
        "KS tests compare full return distributions; Levene tests compare "
        "variances. Standard p-values assume independent observations. Daily "
        "financial returns can be volatility-clustered, so treat these as "
        "indicative and interpret them with the graph and effect sizes."
    )

    st.markdown("### 6. Methodology and limitations")
    with st.expander("Data, calculations and limitations"):
        st.markdown(METHOD_NOTES)

    export = returns[selected_for_risk].join(volumes, how="left", rsuffix="_volume")
    st.download_button(
        "Download selected returns and volume percentiles (CSV)",
        data=export.to_csv().encode("utf-8"),
        file_name="distribution_risk_returns_volume.csv",
        mime="text/csv", key="p4_volcano_download"
    )


def overview():
    """Small dashboard landing-page preview."""
    try:
        returns = load_shared_data()
    except Exception:
        st.caption("Distribution & Risk preview unavailable.")
        return
    rows = []
    for asset in [a for a in ASSET_ORDER if a in returns.columns]:
        pre = risk_statistics(period_slice(returns[[asset]], "2015–2024")[asset])
        post = risk_statistics(period_slice(returns[[asset]], "Since 2025")[asset])
        if pre and post:
            rows.append({
                "Asset": asset,
                "2015–2024 volatility": pre["Annualised volatility (%)"],
                "Since-2025 volatility": post["Annualised volatility (%)"]
            })
    if rows:
        frame = pd.DataFrame(rows).set_index("Asset")
        fig = go.Figure()
        fig.add_bar(name="2015–2024", x=frame.index,
                    y=frame["2015–2024 volatility"], marker_color="#718096")
        fig.add_bar(name="Since 2025", x=frame.index,
                    y=frame["Since-2025 volatility"], marker_color="#E0A82E")
        fig.update_layout(
            barmode="group", height=280,
            title="Annualised volatility: before vs since 2025",
            yaxis_title="Volatility (%)",
            margin=dict(l=5, r=5, t=45, b=5),
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig, width="stretch", key="p4_volcano_overview")
