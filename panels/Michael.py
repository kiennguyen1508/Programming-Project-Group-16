"""Panel 4: compare return/activity distributions and related market risk."""

import json
import logging
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import gaussian_kde

from data_loader import DATA_DIR, get_returns, get_macro, get_macro_changes

try:
    import yfinance as yf
except ImportError:
    yf = None

LOGGER = logging.getLogger(__name__)
SPLIT_DATE = pd.Timestamp("2025-01-01")
YEAR_2026_START = pd.Timestamp("2026-01-01")
TRADING_DAYS = 252
VOLUME_PERCENTILE_WINDOW = 252
MIN_VOLUME_OBSERVATIONS = 60

ASSET_ORDER = ["Gold", "Silver", "Copper", "Oil", "S&P 500"]
ASSET_COLOURS = {
    "Gold": "#E0B252",
    "Silver": "#B8C1CF",
    "Copper": "#C47B4B",
    "Oil": "#4FB39A",
    "S&P 500": "#9C8CFF",
}
PERIODS = ["2015–2024", "2025", "2026 YTD", "Since 2025"]
# Primary Yahoo symbols are the same market series used by the group where possible.
# ETF alternatives are explicitly labelled proxies for trading activity only.
VOLUME_SYMBOLS = {
    "Gold": [("GC=F", "Gold futures volume"), ("GLD", "GLD ETF volume proxy")],
    "Silver": [("SI=F", "Silver futures volume"), ("SLV", "SLV ETF volume proxy")],
    "Copper": [("HG=F", "COMEX copper futures volume"), ("CPER", "CPER ETF volume proxy")],
    "Oil": [("BZ=F", "Brent futures volume"), ("USO", "USO ETF volume proxy")],
    "S&P 500": [("SPY", "SPY ETF volume proxy")],
}
VOLUME_CACHE_CSV = DATA_DIR / "volume_history.csv"
VOLUME_CACHE_SOURCES = DATA_DIR / "volume_history_sources.json"

METHOD_NOTES = """
**Distribution.** Gaussian KDE of daily log returns (%) and trailing volume percentiles (0–100); descriptive, not predictive. Reflected samples reduce boundary bias.

**Volume.** Futures are preferred; GLD, SLV, CPER, USO or SPY are labelled proxies when futures history is insufficient. Trailing percentiles use past observations only; proxies measure related activity, not exact futures volume.

**Risk and tests.** Volatility is annualised with √252; VaR and Expected Shortfall measure historical tail losses. KS/Levene compare pre/post-2025 returns/variance; Holm adjusts for multiple assets. Kupiec/Christoffersen test VaR breaches. Results may be affected by small samples and volatility clustering.

**Limitations.** Tercile comparisons are not causal; 2026 is partial; markets have different calendars; KDE depends on bandwidth; S&P 500 returns exclude dividends. Past results do not predict future returns.
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
            copper = pd.to_numeric(macro["Copper"], errors="coerce")
            log_returns = np.log(copper.where(copper > 0)).diff()
            returns = returns.join(log_returns.rename("Copper"), how="left")

    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns


def download_one_volume_series(symbol, start_date, end_date):
    """Download daily volume from Yahoo Finance."""
    if yf is None:
        LOGGER.warning("Cannot download %s volume: yfinance is not installed.", symbol)
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
        LOGGER.warning("Failed to download volume history for %s.", symbol, exc_info=True)
        return pd.Series(dtype=float, name=symbol)


def has_period_volume_coverage(series, start_date, end_date):
    """Require usable volume observations in each analysis period in range."""
    if not isinstance(series.index, pd.DatetimeIndex) or series.empty:
        return False
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    observed_dates = series.dropna().index
    if (
        observed_dates.min() > start + pd.Timedelta(days=7)
        or observed_dates.max() < end - pd.Timedelta(days=5)
    ):
        return False
    periods = [
        (start, min(end, SPLIT_DATE - pd.Timedelta(days=1))),
        (max(start, SPLIT_DATE), min(end, YEAR_2026_START - pd.Timedelta(days=1))),
        (max(start, YEAR_2026_START), end),
    ]
    checked_period = False
    for period_start, period_end in periods:
        if period_start > period_end:
            continue
        checked_period = True
        period_series = series.loc[period_start:period_end].dropna()
        if (
            len(period_series) < MIN_VOLUME_OBSERVATIONS
            or period_series.index.min() > period_start + pd.Timedelta(days=7)
            or period_series.index.max() < period_end - pd.Timedelta(days=5)
        ):
            return False
    return checked_period


def trailing_volume_percentile(volume):
    """Rank each day's volume within the trailing 252-observation window."""
    observed_volume = volume.dropna()
    percentiles = observed_volume.rolling(
        VOLUME_PERCENTILE_WINDOW,
        min_periods=MIN_VOLUME_OBSERVATIONS,
    ).apply(lambda window: window.rank(pct=True).iloc[-1] * 100)
    return percentiles.reindex(volume.index)


def write_volume_cache(raw_volumes, sources):
    """Atomically replace raw-volume and source cache files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=DATA_DIR, delete=False
        ) as csv_file:
            temporary_paths.append(csv_file.name)
            raw_volumes.rename_axis("Date").to_csv(csv_file)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=DATA_DIR, delete=False
        ) as source_file:
            temporary_paths.append(source_file.name)
            json.dump(sources, source_file, indent=2)
        os.replace(temporary_paths[0], VOLUME_CACHE_CSV)
        os.replace(temporary_paths[1], VOLUME_CACHE_SOURCES)
    except OSError:
        LOGGER.exception("Could not write the trading-volume cache.")
        for temporary_path in temporary_paths:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        raise


def load_volume_data(start_date, end_date):
    """Load raw volumes, retry incomplete sources, and compute trailing ranks."""
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()
    download_start = requested_start - pd.Timedelta(days=370)

    cached_volumes = pd.DataFrame()
    cached_sources = {}
    if VOLUME_CACHE_CSV.exists():
        cached_volumes = pd.read_csv(
            VOLUME_CACHE_CSV, index_col="Date", parse_dates=True
        )
        cached_volumes.index = pd.to_datetime(cached_volumes.index).normalize()
        cached_volumes = cached_volumes.apply(pd.to_numeric, errors="coerce")
        if VOLUME_CACHE_SOURCES.exists():
            with VOLUME_CACHE_SOURCES.open(encoding="utf-8") as source_file:
                cached_sources = json.load(source_file)

    raw_volumes = cached_volumes.copy()
    source_by_asset = dict(cached_sources)
    cache_changed = False
    for asset, candidates in VOLUME_SYMBOLS.items():
        cached_series = cached_volumes.get(asset, pd.Series(dtype=float))
        cached_source = cached_sources.get(asset, "Unavailable")
        if (
            cached_source != "Unavailable"
            and has_period_volume_coverage(
                cached_series, requested_start, requested_end
            )
        ):
            continue

        chosen = pd.Series(dtype=float)
        chosen_source = "Unavailable"
        for symbol, label in candidates:
            candidate = download_one_volume_series(
                symbol, str(download_start.date()), str(requested_end.date())
            )
            if has_period_volume_coverage(candidate, requested_start, requested_end):
                chosen = candidate
                chosen_source = label
                break

        if source_by_asset.get(asset) != chosen_source:
            cache_changed = True
        source_by_asset[asset] = chosen_source
        if not chosen.empty:
            cache_changed = True
            if asset not in raw_volumes:
                raw_volumes[asset] = np.nan
            raw_volumes = raw_volumes.reindex(
                raw_volumes.index.union(chosen.index)
            )
            raw_volumes.loc[chosen.index, asset] = chosen

    if not raw_volumes.empty:
        raw_volumes = raw_volumes.sort_index()
        if cache_changed:
            write_volume_cache(raw_volumes, source_by_asset)
        percentiles = raw_volumes.apply(trailing_volume_percentile)
        return percentiles.loc[requested_start:requested_end], source_by_asset
    return pd.DataFrame(), source_by_asset


def clean_series(series):
    """Convert values to finite numbers and omit missing observations."""
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


def make_joint_distribution_3d(returns, volumes, distribution_specs, bandwidth=1.0):
    """
    Create 3D KDE surfaces for one or two asset-period distributions.

    The axes represent daily return, relative trading-volume percentile, and
    estimated joint probability density. Comparisons use separate scenes.
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
        has_enough_observations = len(data) >= 50
        has_return_variation = data["return"].std() > 0
        has_volume_variation = data["volume_percentile"].std() > 0
        if (
            has_enough_observations
            and has_return_variation
            and has_volume_variation
        ):
            pairs.append((asset, period, data))

    if not pairs:
        return None, None, (
            "Not enough aligned daily-return and trading-volume observations. "
            "Try another asset or a longer period."
        )

    # Use the same X and Y grid for every selected distribution.
    pooled_returns = np.concatenate(
        [
            data["return"].to_numpy(dtype=float) * 100
            for _, _, data in pairs
        ]
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
    density_surfaces = []
    for col, (asset, period, data) in enumerate(pairs, start=1):
        observations = data[["return", "volume_percentile"]].to_numpy(dtype=float)
        observations[:, 0] *= 100

        try:
            reflected_observations = np.concatenate(
                [
                    observations,
                    observations * np.array([1.0, -1.0]),
                    observations * np.array([1.0, -1.0])
                    + np.array([0.0, 200.0]),
                ]
            )
            kde = gaussian_kde(
                reflected_observations.T,
                bw_method=lambda fitted: fitted.scotts_factor() * bandwidth,
            )
            positions = np.vstack([xx.ravel(), yy.ravel()])
            zz = kde(positions).reshape(xx.shape) * 3
        except (np.linalg.LinAlgError, ValueError):
            continue

        surfaces_added += 1
        density_surfaces.append((asset, period, zz))
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
                    "Daily log return: %{x:.2f}%<br>"
                    "Trading-volume percentile: %{y:.0f}<br>"
                    "Joint probability density: %{z:.5f}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )

    if surfaces_added == 0:
        return None, None, (
            "The joint distributions could not be estimated from the available observations."
        )

    density_max = max(float(np.nanmax(density)) for _, _, density in density_surfaces)
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
            title="Daily log return (%)",
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
            range=[0, density_max * 1.05],
            gridcolor="rgba(150,160,180,0.22)",
        ),
        aspectmode="manual",
        aspectratio=dict(x=1.35, y=1.0, z=0.72),
        camera=dict(eye=dict(x=1.55, y=1.65, z=1.0)),
        bgcolor="rgba(0,0,0,0)",
    )

    contour_fig = make_subplots(
        rows=1,
        cols=len(density_surfaces),
        subplot_titles=[
            f"{asset} · {period}" for asset, period, _ in density_surfaces
        ],
        horizontal_spacing=0.08,
    )
    for col, (asset, period, density) in enumerate(density_surfaces, start=1):
        contour_fig.add_trace(
            go.Contour(
                x=x_grid,
                y=y_grid,
                z=density,
                name=f"{asset} · {period}",
                colorscale="Turbo",
                zmin=0,
                zmax=density_max,
                ncontours=20,
                showscale=col == 1,
                colorbar=dict(title="Density"),
                hovertemplate=(
                    f"<b>{asset} · {period}</b><br>"
                    "Daily log return: %{x:.2f}%<br>"
                    "Volume percentile: %{y:.0f}<br>"
                    "Joint density: %{z:.5f}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
    contour_fig.update_layout(
        title="Joint density contours (shared color scale)",
        height=440,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE2EC"),
    )
    contour_fig.update_xaxes(title_text="Daily return (%)")
    contour_fig.update_yaxes(title_text="Volume percentile", range=[0, 100])
    return fig, contour_fig, None


def max_drawdown_from_log_returns(log_returns):
    r = clean_series(log_returns)
    if r.empty:
        return np.nan
    wealth = np.exp(r.cumsum())
    return float((wealth / wealth.cummax() - 1).min())


def risk_statistics(series):
    """Calculate the daily downside-risk summary for one asset period."""
    r = clean_series(series)
    if len(r) < 10:
        return None
    q05 = float(r.quantile(0.05))
    tail = r[r <= q05]
    return {
        "Observations": int(len(r)),
        "Mean daily return (%)": float(r.mean() * 100),
        "Annualised volatility (%)": float(
            r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100
        ),
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
            result = risk_statistics(subset[asset]) if asset in subset else None
            if result:
                rows.append({"Asset": asset, "Period": period, **result})
    return pd.DataFrame(rows)


def activity_relationship_table(returns, volumes, assets):
    """Compare absolute returns in low- and high-volume terciles by period."""
    rows = []
    for asset in assets:
        if asset not in returns or asset not in volumes:
            continue

        row = {"Asset": asset}
        period_gaps = {}
        for period, suffix in (("2015–2024", "2015–2024"), ("Since 2025", "Since 2025")):
            aligned = pd.concat(
                [
                    period_slice(returns[[asset]], period)[asset].rename("return"),
                    period_slice(volumes[[asset]], period)[asset].rename("volume"),
                ],
                axis=1,
            ).dropna()
            if aligned.empty or aligned["volume"].nunique() < 3:
                continue

            low_cut, high_cut = aligned["volume"].quantile([1 / 3, 2 / 3])
            if low_cut >= high_cut:
                continue
            low_returns = aligned.loc[
                aligned["volume"] <= low_cut, "return"
            ].abs()
            high_returns = aligned.loc[
                aligned["volume"] >= high_cut, "return"
            ].abs()
            if len(low_returns) < 30 or len(high_returns) < 30:
                continue

            low_mean = low_returns.mean() * 100
            high_mean = high_returns.mean() * 100
            gap = high_mean - low_mean
            row[f"{suffix} low-volume |return| (%)"] = low_mean
            row[f"{suffix} high-volume |return| (%)"] = high_mean
            row[f"{suffix} high-low gap (pp)"] = gap
            row[f"{suffix} low/high observations"] = (
                f"{len(low_returns)}/{len(high_returns)}"
            )
            period_gaps[suffix] = gap

        if "2015–2024" in period_gaps and "Since 2025" in period_gaps:
            row["Change in gap (pp)"] = (
                period_gaps["Since 2025"] - period_gaps["2015–2024"]
            )
        if len(row) > 1:
            rows.append(row)

    table = pd.DataFrame(rows)
    if "Change in gap (pp)" in table and "Gold" in table["Asset"].values:
        peer_changes = table.loc[
            table["Asset"] != "Gold", "Change in gap (pp)"
        ].dropna()
        gold_change = table.loc[
            table["Asset"] == "Gold", "Change in gap (pp)"
        ].iloc[0]
        if not peer_changes.empty and pd.notna(gold_change):
            table["Gold minus peer-average change (pp)"] = np.nan
            table.loc[
                table["Asset"] == "Gold", "Gold minus peer-average change (pp)"
            ] = gold_change - peer_changes.mean()
    return table


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


def holm_adjust(p_values):
    """Apply Holm's family-wise error correction to a sequence of p-values."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def tests_table(returns, assets):
    """Compare pre-2025 and post-2024 distributions for each selected asset."""
    before = period_slice(returns, "2015–2024")
    after = period_slice(returns, "Since 2025")
    rows = []
    for asset in assets:
        x, y = clean_series(before[asset]), clean_series(after[asset])
        if len(x) >= 30 and len(y) >= 20:
            ks = stats.ks_2samp(x, y)
            lev = stats.levene(x, y, center="median")
            rows.append(
                {
                    "Asset": asset,
                    "KS statistic": ks.statistic,
                    "KS p-value": ks.pvalue,
                    "Levene statistic": lev.statistic,
                    "Levene p-value": lev.pvalue,
                    "Post/pre volatility ratio": (
                        y.std(ddof=1) / x.std(ddof=1)
                    ),
                    "Pre-2025 n": len(x),
                    "Post-2024 n": len(y),
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["KS p-value (Holm)"] = holm_adjust(table["KS p-value"])
        table["Levene p-value (Holm)"] = holm_adjust(
            table["Levene p-value"]
        )
    return table


def bernoulli_log_likelihood(successes, total, probability):
    """Return a Bernoulli log likelihood without undefined zero-count terms."""
    failures = total - successes
    if (successes and probability <= 0) or (failures and probability >= 1):
        return -np.inf
    likelihood = 0.0
    if successes:
        likelihood += successes * np.log(probability)
    if failures:
        likelihood += failures * np.log1p(-probability)
    return float(likelihood)


def var_backtest_tests(breaches, expected_rate=0.05):
    """Return Kupiec coverage and Christoffersen independence test results."""
    breach_values = np.asarray(breaches, dtype=bool)
    count = int(breach_values.sum())
    total = len(breach_values)
    observed_rate = count / total

    null_likelihood = bernoulli_log_likelihood(
        count, total, expected_rate
    )
    observed_likelihood = bernoulli_log_likelihood(
        count, total, observed_rate
    )
    kupiec_lr = max(0.0, 2 * (observed_likelihood - null_likelihood))

    previous = breach_values[:-1]
    current = breach_values[1:]
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())
    transitions = n00 + n01 + n10 + n11
    overall_probability = (n01 + n11) / transitions
    probability_after_nonbreach = n01 / (n00 + n01) if n00 + n01 else 0.0
    probability_after_breach = n11 / (n10 + n11) if n10 + n11 else 0.0
    independence_null = bernoulli_log_likelihood(
        n01 + n11, transitions, overall_probability
    )
    independence_alternative = (
        bernoulli_log_likelihood(n01, n00 + n01, probability_after_nonbreach)
        + bernoulli_log_likelihood(n11, n10 + n11, probability_after_breach)
    )
    christoffersen_lr = max(
        0.0, 2 * (independence_alternative - independence_null)
    )
    return {
        "Kupiec LR": kupiec_lr,
        "Kupiec p-value": stats.chi2.sf(kupiec_lr, df=1),
        "Christoffersen LR": christoffersen_lr,
        "Christoffersen p-value": stats.chi2.sf(christoffersen_lr, df=1),
    }


def var_breach_table(returns, assets):
    """Test post-2024 returns against each asset's 2015–2024 95% VaR."""
    before = period_slice(returns, "2015–2024")
    after = period_slice(returns, "Since 2025")
    rows = []
    for asset in assets:
        base, later = clean_series(before[asset]), clean_series(after[asset])
        if len(base) >= 100 and len(later) >= 20:
            threshold = base.quantile(0.05)
            breach = later <= threshold
            rows.append(
                {
                    "Asset": asset,
                    "Historical 95% VaR loss (%)": -threshold * 100,
                    "Post-2024 breach rate (%)": breach.mean() * 100,
                    "Expected breach rate (%)": 5.0,
                    "Breach days": int(breach.sum()),
                    "Post-2024 observations": len(later),
                    **var_backtest_tests(breach),
                }
            )
    return pd.DataFrame(rows)


def format_p_value(value):
    """Keep very small p-values from appearing as zero in the results tables."""
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def render():
    st.subheader("Distribution & Risk")
    st.caption(
        "Research question: How has the relationship between daily price changes "
        "and trading activity evolved since 2025, and was the change distinct for Gold?"
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

    st.markdown("### 1. Interactive 3D joint probability distribution")
    st.write("Explore return and trading-activity distributions; optionally compare a second asset or period.")

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
            "Analysis period",
            PERIODS[:3],
            default=PERIODS[0],
            key="p4_primary_period",
        ) or PERIODS[0]
    with c3:
        bandwidth = st.select_slider(
            "KDE bandwidth",
            options=[0.65, 0.8, 1.0, 1.2, 1.4],
            value=1.0,
            format_func=lambda x: f"{x:.2f}×",
            key="p4_distribution_bandwidth",
        )

    compare_enabled = st.checkbox(
        "Compare with a second asset or period",
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
                "Comparison period",
                PERIODS[:3],
                default=PERIODS[1],
                key="p4_compare_period",
            ) or PERIODS[1]
        distribution_specs.append((compare_asset, compare_period))

    chosen_assets = list(dict.fromkeys(asset for asset, _ in distribution_specs))
    chosen_periods = list(dict.fromkeys(period for _, period in distribution_specs))

    if not volumes.empty:
        st.caption("Volume sources — " + " | ".join(
            f"**{asset}:** {volume_sources.get(asset, 'Unavailable')}"
            for asset in chosen_assets
        ))

    missing_volume = [
        asset
        for asset in chosen_assets
        if asset not in volumes.columns
        or volume_sources.get(asset, "Unavailable") == "Unavailable"
    ]
    if missing_volume:
        st.warning(
            "Volume unavailable for: " + ", ".join(missing_volume)
            + ". Check the Yahoo Finance symbols or remove the unavailable asset."
        )
    elif chosen_assets:
        fig, contour_fig, error = make_joint_distribution_3d(
            returns, volumes, distribution_specs, bandwidth
        )
        if error:
            st.warning(error)
        else:
            chart_column, contour_column = st.columns([1.25, 1], gap="medium")
            with chart_column:
                st.plotly_chart(
                    fig,
                    width="stretch",
                    key="p4_joint_distribution_3d",
                    config={"scrollZoom": True, "displaylogo": False},
                )
            with contour_column:
                st.plotly_chart(
                    contour_fig,
                    width="stretch",
                    key="p4_joint_distribution_contours",
                )

    st.caption(
        "X = daily log return (%); Y = trailing volume percentile (relative activity); "
        "Z = estimated joint density. Contours share a density scale."
    )

    st.markdown("### 2. Risk measures")
    selected_for_risk = chosen_assets or assets
    periods_for_risk = chosen_periods or ["2015–2024", "2025", "2026 YTD"]
    table = risk_table(returns, selected_for_risk, periods_for_risk)
    if not table.empty:
        numeric = [
            c
            for c in table.columns
            if c not in ["Asset", "Period", "Observations"]
        ]
        table[numeric] = table[numeric].round(3)
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.info("Not enough observations for the selected risk statistics.")
    st.caption("VaR and Expected Shortfall are daily losses; volatility is annualised. Small-sample tail estimates are less stable.")

    st.markdown("### 3. Historical 95% VaR backtest")
    breaches = var_breach_table(returns, selected_for_risk)
    if not breaches.empty:
        breaches_display = breaches.copy().round(3)
        for column in ("Kupiec p-value", "Christoffersen p-value"):
            breaches_display[column] = breaches[column].map(format_p_value)
        st.dataframe(breaches_display, hide_index=True, width="stretch")
    else:
        st.info("Insufficient pre-2025 or post-2024 history to backtest 95% VaR.")
    st.caption(
        "The 2015–2024 5th-percentile loss threshold is tested against 2025 onwards. "
        "Kupiec tests coverage; Christoffersen tests breach independence."
    )

    st.markdown("### 4. Rolling volatility")
    window = st.select_slider(
        "Rolling volatility window", [21, 30, 60, 90],
        value=30, format_func=lambda d: f"{d} trading days",
        key="p4_distribution_rolling_window")
    st.plotly_chart(
        make_rolling_volatility(returns, selected_for_risk, window),
        width="stretch", key="p4_distribution_rolling"
    )

    st.markdown("### 5. Relationship between trading activity and returns")
    activity = activity_relationship_table(
        returns, volumes, assets
    )
    if not activity.empty:
        gold_peer_column = "Gold minus peer-average change (pp)"
        gold_peer_change = None
        if gold_peer_column in activity:
            gold_values = activity.loc[
                activity["Asset"] == "Gold", gold_peer_column
            ].dropna()
            if not gold_values.empty:
                gold_peer_change = gold_values.iloc[0]
        activity_display = activity.drop(columns=[gold_peer_column], errors="ignore")
        numeric = [
            column
            for column in activity_display.columns
            if "observations" not in column.lower() and column != "Asset"
        ]
        activity_display[numeric] = activity_display[numeric].round(3)
        st.dataframe(activity_display, hide_index=True, width="stretch")
        if gold_peer_change is not None:
            st.metric(
                "Gold minus peer-average change (pp)",
                f"{gold_peer_change:+.3f}",
            )
    else:
        st.info("Insufficient aligned return and volume data for the comparison.")
    st.caption("The high-minus-low return gap and Gold-versus-peer change are descriptive, not causal estimates.")

    st.markdown("### 6. Statistical tests for distribution changes")
    tests = tests_table(returns, selected_for_risk)
    if not tests.empty:
        tests_display = tests.drop(
            columns=["KS p-value", "Levene p-value"], errors="ignore"
        )
        for column in ("KS p-value (Holm)", "Levene p-value (Holm)"):
            tests_display[column] = tests[column].map(format_p_value)
        st.dataframe(tests_display.round(4), hide_index=True, width="stretch")
    else:
        st.info("Insufficient observations for the pre-2025 versus post-2024 tests.")
    st.caption(
        "KS compares distributions; Levene compares variances. Holm-adjusted "
        "p-values account for selected assets. Results are indicative: returns "
        "may exhibit volatility clustering."
    )

    st.markdown("### 7. Methodology and limitations")
    with st.expander("Data, calculations and limitations"):
        st.markdown(METHOD_NOTES)

    return_export = returns[selected_for_risk].add_suffix("_log_return")
    volume_export = volumes.reindex(columns=selected_for_risk).add_suffix(
        "_volume_percentile"
    )
    export = return_export.join(volume_export, how="left")
    st.download_button(
        "Download selected returns and volume percentiles (CSV)",
        data=export.to_csv().encode("utf-8"),
        file_name="distribution_risk_returns_volume.csv",
        mime="text/csv", key="p4_distribution_download"
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
        series = returns[asset]
        pre = risk_statistics(period_slice(series.to_frame(), "2015–2024")[asset])
        post = risk_statistics(period_slice(series.to_frame(), "Since 2025")[asset])
        if pre and post:
            rows.append({
                "Asset": asset,
                "2015–2024 volatility": pre["Annualised volatility (%)"],
                "Since-2025 volatility": post["Annualised volatility (%)"],
            })
    if not rows:
        return
    frame = pd.DataFrame(rows).set_index("Asset")
    fig = go.Figure([
        go.Bar(name="2015–2024", x=frame.index, y=frame["2015–2024 volatility"], marker_color="#718096"),
        go.Bar(name="Since 2025", x=frame.index, y=frame["Since-2025 volatility"], marker_color="#E0A82E"),
    ])
    fig.update_layout(
        barmode="group", height=280, title="Annualised volatility: before vs since 2025",
        yaxis_title="Volatility (%)", margin=dict(l=5, r=5, t=45, b=5),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, width="stretch", key="p4_distribution_overview")
