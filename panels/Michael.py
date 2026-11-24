""" 
Panel 4: Distribution and risk (Michael).

This panel looks at how the returns of the selected assets are distributed and
how their risk changes over time. It combines a 3D joint-distribution chart,
risk measures, trading-volume comparisons and statistical tests.

Data and methods
----------------
Daily log returns come from the shared data loader. Trading volume is downloaded
from Yahoo Finance and matched to each asset where possible. If a futures series
does not have enough usable volume history, the panel uses the relevant ETF as
a volume proxy (for example, GLD for gold and SLV for silver).

The 3D chart uses a Gaussian kernel density estimate (KDE). Daily returns are
shown against the trailing volume percentile, with the estimated joint density
on the vertical axis. Volume percentiles are calculated using the current and
previous observations only, so later volume data is not used to calculate an
earlier date's percentile.

The risk section reports measures such as annualised volatility, Value at Risk,
Expected Shortfall, skewness, kurtosis and maximum drawdown. The panel also
compares activity across lower- and higher-volume days, tests changes in return
distributions and volatility, and checks historical VaR breaches.

A few points to bear in mind
----------------------------
ETF volume is a proxy for activity in the related market, not the exact volume
of the futures contract or index. Futures prices and volume can also be affected
by contract rolls. Statistical tests rely on simplifying assumptions, and
historical results do not predict future returns. Results based on small
samples, especially tail-risk measures, should be treated with caution.

The functions below are kept separate so the data preparation, calculations
and Streamlit display can be followed independently.
"""

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
SPLIT_DATE = pd.Timestamp('2025-01-01')
YEAR_2026_START = pd.Timestamp('2026-01-01')
TRADING_DAYS = 252
VOLUME_PERCENTILE_WINDOW = 252
MIN_VOLUME_OBSERVATIONS = 60
MIN_RELIABLE_SAMPLE = 250

ASSET_ORDER = ['Gold', 'Silver', 'Copper', 'Oil', 'S&P 500']
ASSET_COLOURS = {
    'Gold': '#E0B252',
    'Silver': '#B8C1CF',
    'Copper': '#C47B4B',
    'Oil': '#4FB39A',
    'S&P 500': '#9C8CFF'
}

PERIODS = ['2015–2024', '2025', '2026 YTD', 'Since 2025']

VOLUME_SYMBOLS = {
    'Gold': [('GC=F', 'Gold futures volume'), ('GLD', 'GLD ETF volume proxy')],
    'Silver': [('SI=F', 'Silver futures volume'), ('SLV', 'SLV ETF volume proxy')],
    'Copper': [('HG=F', 'COMEX copper futures volume'), ('CPER', 'CPER ETF volume proxy')],
    'Oil': [('BZ=F', 'Brent futures volume'), ('USO', 'USO ETF volume proxy')],
    'S&P 500': [('SPY', 'SPY ETF volume proxy')]
}

VOLUME_CACHE_CSV = DATA_DIR / 'volume_history.csv'
VOLUME_CACHE_SOURCES = DATA_DIR / 'volume_history_sources.json'

METHOD_NOTES = """
**Joint probability-density surface.** The surface is a two-dimensional
Gaussian kernel density estimate (KDE) of daily log returns and trading
activity. X represents daily log returns in percent; Y represents each asset's
volume percentile within the trailing 252-observation window ending on that
date (using available history at the start of the sample); Z represents
estimated joint probability density. Reflected KDE samples
reduce boundary bias at percentiles 0 and 100. Higher-density regions indicate
return and activity combinations that are more concentrated in the observed
sample. This is a descriptive joint distribution, not a forecast or evidence
of causation.

**Volume treatment.** Futures volume is used where Yahoo Finance provides
enough observations. Otherwise the panel uses a named ETF proxy (GLD, SLV,
CPER, USO or SPY). Volume is converted to a percentile within each instrument
so that 90 means unusually high activity for that instrument, not the same
number of contracts or shares. Percentiles use only the current and preceding
251 available observations, avoiding look-ahead from later volume data. The
initial percentiles use a shorter window until sufficient history is available.
ETF proxy volume is a measure of related-market activity; it is not the exact
volume of the futures or index series whose returns are plotted. Continuous
futures contracts can show volume disruptions around contract rolls.

**Risk.** Annualised volatility = daily standard deviation × √252. 95% VaR is
the fifth-percentile loss threshold; Expected Shortfall is the average loss
among returns at or below that threshold. Skewness measures asymmetry and
excess kurtosis measures tail heaviness. Maximum drawdown is calculated from
the cumulative wealth path reconstructed from log returns.

**Tests.** KS compares the pre-2025 and post-2024 return distributions;
Levene compares their variances. P-values are Holm-adjusted across the selected
assets. Standard p-values assume independent observations, while financial
returns can exhibit volatility clustering. The VaR backtest reports Kupiec
coverage and Christoffersen independence tests; these also rely on simplifying
distributional assumptions. Interpret all tests alongside sample sizes, effect
sizes and visual evidence.

**Activity comparison.** Mean absolute daily log returns are compared between
low- and high-volume terciles in 2015–2024 and since 2025. The change in the
high-minus-low gap for Gold is also compared with the average change for the
available peer assets. This is a descriptive difference-in-differences summary,
not a causal estimate.

**Limitations.** Rows with fewer than 250 observations are flagged; tail
statistics are less stable for these samples. The activity comparison uses
empirical volume terciles and does not establish causality. 2026 is a partial
year. Different markets have different
trading hours and holidays. Futures can be affected by contract rolls. KDE
shape depends on bandwidth and sample size. S&P 500 index returns exclude
dividends. These historical patterns do not predict future returns.
"""


@st.cache_data(show_spinner=False, max_entries=10)
def load_shared_data():
    """Load shared returns and add Copper returns if they are not already present."""
    returns = get_returns().copy()
    macro = get_macro().copy()
    macro_changes = get_macro_changes().copy()

    for frame in (returns, macro, macro_changes):
        frame.index = pd.to_datetime(frame.index)
        if frame.index.tz is not None:
            frame.index = frame.index.tz_localize(None)
        frame.sort_index(inplace=True)

    if 'Copper' not in returns.columns:
        if 'Copper' in macro_changes.columns:
            returns = returns.join(macro_changes['Copper'].rename('Copper'), how='left')
        elif 'Copper' in macro.columns:
            copper = pd.to_numeric(macro['Copper'], errors='coerce')
            log_returns = np.log(copper.where(copper > 0)).diff()
            returns = returns.join(log_returns.rename('Copper'), how='left')
    returns = returns.apply(pd.to_numeric, errors='coerce')
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns


def download_one_volume_series(symbol, start_date, end_date):
    """Download daily volume for one Yahoo Finance symbol."""
    if yf is None:
        LOGGER.warning('Cannot download %s volume: yfinance is not installed.', symbol)
        return pd.Series(dtype=float, name=symbol)
    try:
        hist = yf.Ticker(symbol).history(start=start_date, end=end_date, auto_adjust=False, actions=False)
        if hist.empty or 'Volume' not in hist.columns:
            return pd.Series(dtype=float, name=symbol)
        volume = pd.to_numeric(hist['Volume'], errors='coerce')
        idx = pd.to_datetime(hist.index)
        if getattr(idx, 'tz', None) is not None:
            idx = idx.tz_localize(None)
        volume.index = idx.normalize()
        volume = volume[volume > 0].groupby(level=0).last()
        volume.name = symbol
        return volume
    except Exception:
        LOGGER.warning('Failed to download volume history for %s.', symbol, exc_info=True)
        return pd.Series(dtype=float, name=symbol)


def has_period_volume_coverage(series, start_date, end_date):
    """Check that a volume series has enough observations across the requested periods."""
    if not isinstance(series.index, pd.DatetimeIndex) or series.empty:
        return False
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    observed_dates = series.dropna().index
    if observed_dates.min() > start + pd.Timedelta(days=7) or observed_dates.max() < end - pd.Timedelta(days=5):
        return False
    periods = [(start, min(end, SPLIT_DATE - pd.Timedelta(days=1))), (max(start, SPLIT_DATE), min(end, YEAR_2026_START - pd.Timedelta(days=1))), (max(start, YEAR_2026_START), end)]
    checked_period = False
    for period_start, period_end in periods:
        if period_start > period_end:
            continue
        checked_period = True
        period_series = series.loc[period_start:period_end].dropna()
        if len(period_series) < MIN_VOLUME_OBSERVATIONS or period_series.index.min() > period_start + pd.Timedelta(days=7) or period_series.index.max() < period_end - pd.Timedelta(days=5):
            return False
    return checked_period


def trailing_volume_percentile(volume):
    """Compare each volume observation with its trailing year of activity."""
    observed_volume = volume.dropna()
    # I use a trailing window so a date's percentile is not affected by later trading activity.
    percentiles = observed_volume.rolling(VOLUME_PERCENTILE_WINDOW, min_periods=MIN_VOLUME_OBSERVATIONS).apply(lambda window: window.rank(pct=True).iloc[-1] * 100)
    return percentiles.reindex(volume.index)


def write_volume_cache(raw_volumes, sources):
    """Save the downloaded volume data and its source information to the local cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=DATA_DIR, delete=False) as csv_file:
            temporary_paths.append(csv_file.name)
            raw_volumes.rename_axis('Date').to_csv(csv_file)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=DATA_DIR, delete=False) as source_file:
            temporary_paths.append(source_file.name)
            json.dump(sources, source_file, indent=2)
        os.replace(temporary_paths[0], VOLUME_CACHE_CSV)
        os.replace(temporary_paths[1], VOLUME_CACHE_SOURCES)
    except OSError:
        LOGGER.exception('Could not write the trading-volume cache.')
        for temporary_path in temporary_paths:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        raise


def load_volume_data(start_date, end_date):
    """Load volume history, reusing the local cache when it has enough coverage."""
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()
    download_start = requested_start - pd.Timedelta(days=370)
    cached_volumes = pd.DataFrame()
    cached_sources = {}
    if VOLUME_CACHE_CSV.exists():
        cached_volumes = pd.read_csv(VOLUME_CACHE_CSV, index_col='Date', parse_dates=True)
        cached_volumes.index = pd.to_datetime(cached_volumes.index).normalize()
        cached_volumes = cached_volumes.apply(pd.to_numeric, errors='coerce')
        if VOLUME_CACHE_SOURCES.exists():
            with VOLUME_CACHE_SOURCES.open(encoding='utf-8') as source_file:
                cached_sources = json.load(source_file)
    # I keep the raw volumes in a cache so the panel does not need to download them on every rerun.
    raw_volumes = cached_volumes.copy()
    source_by_asset = dict(cached_sources)
    cache_changed = False
    # Prefer futures volume, then fall back to the related ETF if coverage is poor.
    for asset, candidates in VOLUME_SYMBOLS.items():
        cached_series = cached_volumes.get(asset, pd.Series(dtype=float))
        cached_source = cached_sources.get(asset, 'Unavailable')
        if cached_source != 'Unavailable' and has_period_volume_coverage(cached_series, requested_start, requested_end):
            continue
        chosen = pd.Series(dtype=float)
        chosen_source = 'Unavailable'
        for symbol, label in candidates:
            candidate = download_one_volume_series(symbol, str(download_start.date()), str(requested_end.date()))
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
            raw_volumes = raw_volumes.reindex(raw_volumes.index.union(chosen.index))
            raw_volumes.loc[chosen.index, asset] = chosen
    if not raw_volumes.empty:
        raw_volumes = raw_volumes.sort_index()
        if cache_changed:
            write_volume_cache(raw_volumes, source_by_asset)
        percentiles = raw_volumes.apply(trailing_volume_percentile)
        return (percentiles.loc[requested_start:requested_end], source_by_asset)
    return (pd.DataFrame(), source_by_asset)


def clean_series(series):
    """Convert a series to numeric values and remove missing or infinite observations."""
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()


def period_slice(data, period):
    """Return the observations that fall within the selected analysis period."""
    if period == '2015–2024':
        return data.loc[data.index < SPLIT_DATE]
    if period == '2025':
        return data.loc[(data.index >= SPLIT_DATE) & (data.index < YEAR_2026_START)]
    if period == '2026 YTD':
        return data.loc[data.index >= YEAR_2026_START]
    if period == 'Since 2025':
        return data.loc[data.index >= SPLIT_DATE]
    return data.iloc[0:0]
def make_joint_distribution_3d(returns, volumes, distribution_specs, bandwidth=1.0):
    """Create the 3D joint-density surfaces and their matching contour plots."""
    # Each point in the surface is a return and volume-percentile observation for the same date.
    pairs = []
    for asset, period in distribution_specs:
        if asset not in returns.columns or asset not in volumes.columns:
            continue
        r = period_slice(returns[[asset]], period)[asset]
        v = volumes[asset].reindex(r.index)
        data = pd.concat([r.rename('return'), v.rename('volume_percentile')], axis=1).dropna()
        has_enough_observations = len(data) >= 50
        has_return_variation = data['return'].std() > 0
        has_volume_variation = data['volume_percentile'].std() > 0
        if has_enough_observations and has_return_variation and has_volume_variation:
            pairs.append((asset, period, data))
    if not pairs:
        return (None, None, 'Not enough aligned daily-return and trading-volume observations. Try another asset or a longer period.')
    pooled_returns = np.concatenate([data['return'].to_numpy(dtype=float) * 100 for _, _, data in pairs])
    x_lo, x_hi = np.nanpercentile(pooled_returns, [0.5, 99.5])
    x_pad = max((x_hi - x_lo) * 0.15, 0.2)
    x_grid = np.linspace(x_lo - x_pad, x_hi + x_pad, 90)
    y_grid = np.linspace(0, 100, 70)
    xx, yy = np.meshgrid(x_grid, y_grid)
    fig = make_subplots(rows=1, cols=len(pairs), specs=[[{'type': 'scene'} for _ in pairs]], subplot_titles=[f'{asset} · {period}' for asset, period, _ in pairs], horizontal_spacing=0.04)
    surfaces_added = 0
    density_surfaces = []
    for col, (asset, period, data) in enumerate(pairs, start=1):
        observations = data[['return', 'volume_percentile']].to_numpy(dtype=float)
        observations[:, 0] *= 100
        try:
            # Reflect the volume percentiles at 0 and 100 to reduce KDE edge bias.
            reflected_observations = np.concatenate([observations, observations * np.array([1.0, -1.0]), observations * np.array([1.0, -1.0]) + np.array([0.0, 200.0])])
            kde = gaussian_kde(reflected_observations.T, bw_method=lambda fitted: fitted.scotts_factor() * bandwidth)
            positions = np.vstack([xx.ravel(), yy.ravel()])
            zz = kde(positions).reshape(xx.shape) * 3
        except (np.linalg.LinAlgError, ValueError):
            continue
        surfaces_added += 1
        density_surfaces.append((asset, period, zz))
        label = f'{asset} · {period}'
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz, name=label, colorscale='Turbo', showscale=False, opacity=1.0, lighting=dict(ambient=0.52, diffuse=0.88, specular=0.45, roughness=0.42), contours=dict(z=dict(show=True, usecolormap=False, color='rgba(255,255,255,0.48)', width=1)), hovertemplate=f'<b>{label}</b><br>Daily log return: %{{x:.2f}}%<br>Trading-volume percentile: %{{y:.0f}}<br>Joint probability density: %{{z:.5f}}<extra></extra>'), row=1, col=col)
    if surfaces_added == 0:
        return (None, None, 'The joint distributions could not be estimated from the available observations.')
    density_max = max((float(np.nanmax(density)) for _, _, density in density_surfaces))
    fig.update_layout(title=dict(text='Joint probability distribution: daily returns × trading activity', x=0.02), height=720 if len(pairs) == 1 else 760, margin=dict(l=0, r=0, t=90, b=10), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#DCE2EC'))
    fig.update_scenes(xaxis=dict(title='Daily log return (%)', range=[x_lo - x_pad, x_hi + x_pad], gridcolor='rgba(150,160,180,0.22)', zeroline=True), yaxis=dict(title='Trading volume (within-asset percentile)', range=[0, 100], gridcolor='rgba(150,160,180,0.22)'), zaxis=dict(title='Probability density', range=[0, density_max * 1.05], gridcolor='rgba(150,160,180,0.22)'), aspectmode='manual', aspectratio=dict(x=1.35, y=1.0, z=0.72), camera=dict(eye=dict(x=1.55, y=1.65, z=1.0)), bgcolor='rgba(0,0,0,0)')
    contour_fig = make_subplots(rows=1, cols=len(density_surfaces), subplot_titles=[f'{asset} · {period}' for asset, period, _ in density_surfaces], horizontal_spacing=0.08)
    for col, (asset, period, density) in enumerate(density_surfaces, start=1):
        contour_fig.add_trace(go.Contour(x=x_grid, y=y_grid, z=density, name=f'{asset} · {period}', colorscale='Turbo', zmin=0, zmax=density_max, ncontours=20, showscale=col == 1, colorbar=dict(title='Density'), hovertemplate=f'<b>{asset} · {period}</b><br>Daily log return: %{{x:.2f}}%<br>Volume percentile: %{{y:.0f}}<br>Joint density: %{{z:.5f}}<extra></extra>'), row=1, col=col)
    contour_fig.update_layout(title='Joint density contours (shared color scale)', height=440, margin=dict(l=20, r=20, t=80, b=20), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#DCE2EC'))
    contour_fig.update_xaxes(title_text='Daily return (%)')
    contour_fig.update_yaxes(title_text='Volume percentile', range=[0, 100])
    return (fig, contour_fig, None)


def max_drawdown_from_log_returns(log_returns):
    """Calculate the largest peak-to-trough fall from a series of log returns."""
    r = clean_series(log_returns)
    if r.empty:
        return np.nan
    wealth = np.exp(r.cumsum())
    return float((wealth / wealth.cummax() - 1).min())


def risk_statistics(series):
    """Calculate the daily and annualised risk measures shown in the panel."""
    r = clean_series(series)
    if len(r) < 10:
        return None
    q05 = float(r.quantile(0.05))
    tail = r[r <= q05]
    return {'Observations': int(len(r)), 'Mean daily return (%)': float(r.mean() * 100), 'Annualised volatility (%)': float(r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100), '95% VaR (%)': float(-q05 * 100), '95% Expected Shortfall (%)': float(-tail.mean() * 100), 'Skewness': float(stats.skew(r, bias=False)), 'Excess kurtosis': float(stats.kurtosis(r, fisher=True, bias=False)), 'Maximum drawdown (%)': float(max_drawdown_from_log_returns(r) * 100)}


def risk_table(returns, assets, periods):
    """Put the risk statistics for the selected assets and periods into one table."""
    rows = []
    for period in periods:
        subset = period_slice(returns, period)
        for asset in assets:
            if asset in subset:
                result = risk_statistics(subset[asset])
                if result:
                    rows.append({'Asset': asset, 'Period': period, **result, 'Sample note': 'Limited sample' if result['Observations'] < MIN_RELIABLE_SAMPLE else ''})
    return pd.DataFrame(rows)


def activity_relationship_table(returns, volumes, assets):
    """Compare absolute returns on low- and high-volume days."""
    rows = []
    for asset in assets:
        if asset not in returns or asset not in volumes:
            continue
        row = {'Asset': asset}
        period_gaps = {}
        for period, suffix in (('2015–2024', '2015–2024'), ('Since 2025', 'Since 2025')):
            aligned = pd.concat([period_slice(returns[[asset]], period)[asset].rename('return'), period_slice(volumes[[asset]], period)[asset].rename('volume')], axis=1).dropna()
            if aligned.empty or aligned['volume'].nunique() < 3:
                continue
            low_cut, high_cut = aligned['volume'].quantile([1 / 3, 2 / 3])
            if low_cut >= high_cut:
                continue
            low_returns = aligned.loc[aligned['volume'] <= low_cut, 'return'].abs()
            high_returns = aligned.loc[aligned['volume'] >= high_cut, 'return'].abs()
            if len(low_returns) < 30 or len(high_returns) < 30:
                continue
            low_mean = low_returns.mean() * 100
            high_mean = high_returns.mean() * 100
            gap = high_mean - low_mean
            row[f'{suffix} low-volume |return| (%)'] = low_mean
            row[f'{suffix} high-volume |return| (%)'] = high_mean
            row[f'{suffix} high-low gap (pp)'] = gap
            row[f'{suffix} low/high observations'] = f'{len(low_returns)}/{len(high_returns)}'
            period_gaps[suffix] = gap
        if '2015–2024' in period_gaps and 'Since 2025' in period_gaps:
            row['Change in gap (pp)'] = period_gaps['Since 2025'] - period_gaps['2015–2024']
        if len(row) > 1:
            rows.append(row)
    table = pd.DataFrame(rows)
    if 'Change in gap (pp)' in table and 'Gold' in table['Asset'].values:
        peer_changes = table.loc[table['Asset'] != 'Gold', 'Change in gap (pp)'].dropna()
        gold_change = table.loc[table['Asset'] == 'Gold', 'Change in gap (pp)'].iloc[0]
        if not peer_changes.empty and pd.notna(gold_change):
            table['Gold minus peer-average change (pp)'] = np.nan
            table.loc[table['Asset'] == 'Gold', 'Gold minus peer-average change (pp)'] = gold_change - peer_changes.mean()
    return table


def make_rolling_volatility(returns, assets, window=30):
    """Create the rolling annualised-volatility chart for the selected assets."""
    fig = go.Figure()
    for asset in assets:
        rolling = returns[asset].rolling(window, min_periods=max(15, window // 2)).std() * np.sqrt(TRADING_DAYS) * 100
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling, mode='lines', name=asset, line=dict(color=ASSET_COLOURS.get(asset, '#A0AAB8'), width=2)))
    fig.add_vline(x=SPLIT_DATE, line_dash='dash', line_color='#E0B252', annotation_text='Start of 2025')
    fig.update_layout(title=f'{window}-day rolling annualised volatility', height=380, xaxis_title=None, yaxis_title='Annualised volatility (%)', hovermode='x unified', legend=dict(orientation='h', y=1.12), margin=dict(l=10, r=10, t=60, b=10))
    return fig


def holm_adjust(p_values):
    """Apply the Holm correction to a set of p-values."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate((len(p_values) - np.arange(len(p_values))) * p_values[order])
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted
def tests_table(returns, assets):
    """Compare return distributions and variances before and after 2025."""
    rows = []
    for asset in assets:
        if asset not in returns:
            continue
        before = clean_series(returns.loc[returns.index < SPLIT_DATE, asset])
        after = clean_series(returns.loc[returns.index >= SPLIT_DATE, asset])
        if len(before) < 30 or len(after) < 30:
            continue
        ks = stats.ks_2samp(before, after)
        levene = stats.levene(before, after, center='median')
        rows.append({'Asset': asset, 'Pre-2025 observations': len(before), 'Since-2025 observations': len(after), 'KS statistic': ks.statistic, 'KS p-value': ks.pvalue, 'Levene statistic': levene.statistic, 'Levene p-value': levene.pvalue})
    table = pd.DataFrame(rows)
    if not table.empty:
        table['KS Holm-adjusted p-value'] = holm_adjust(table['KS p-value'])
        table['Levene Holm-adjusted p-value'] = holm_adjust(table['Levene p-value'])
    return table


def bernoulli_log_likelihood(breaches, probability):
    """Calculate the log likelihood for a sequence of VaR breach indicators."""
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    breaches = np.asarray(breaches, dtype=int)
    return float(np.sum(breaches * np.log(probability) + (1 - breaches) * np.log(1 - probability)))


def var_backtest_tests(series, confidence=0.95):
    """Run unconditional coverage and independence tests for historical VaR breaches."""
    r = clean_series(series)
    if len(r) < MIN_RELIABLE_SAMPLE:
        return None
    threshold = r.quantile(1 - confidence)
    breaches = (r < threshold).astype(int).to_numpy()
    n = len(breaches)
    x = int(breaches.sum())
    expected_probability = 1 - confidence
    observed_probability = x / n
    null_ll = bernoulli_log_likelihood(breaches, expected_probability)
    alternative_ll = bernoulli_log_likelihood(breaches, observed_probability)
    coverage_stat = max(0.0, -2 * (null_ll - alternative_ll))
    coverage_p = float(stats.chi2.sf(coverage_stat, 1))
    previous = breaches[:-1]
    current = breaches[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))
    p01 = n01 / (n00 + n01) if n00 + n01 else 0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0
    pooled_probability = (n01 + n11) / max(n - 1, 1)
    independent_ll = bernoulli_log_likelihood(np.r_[np.zeros(n00), np.ones(n01)], p01) + bernoulli_log_likelihood(np.r_[np.zeros(n10), np.ones(n11)], p11)
    pooled_ll = bernoulli_log_likelihood(np.r_[np.zeros(n00 + n10), np.ones(n01 + n11)], pooled_probability)
    independence_stat = max(0.0, -2 * (pooled_ll - independent_ll))
    independence_p = float(stats.chi2.sf(independence_stat, 1))
    return {'Observations': n, 'VaR breaches': x, 'Expected breaches': n * expected_probability, 'Observed breach rate (%)': observed_probability * 100, 'Expected breach rate (%)': expected_probability * 100, 'Kupiec statistic': coverage_stat, 'Kupiec p-value': coverage_p, 'Christoffersen statistic': independence_stat, 'Christoffersen p-value': independence_p}


def var_breach_table(returns, assets, confidence=0.95):
    """Summarise VaR backtest results for each selected asset."""
    rows = []
    for asset in assets:
        if asset not in returns:
            continue
        result = var_backtest_tests(returns[asset], confidence)
        if result:
            rows.append({'Asset': asset, **result})
    return pd.DataFrame(rows)


def render():
    """Draw Panel 4 inside the Streamlit dashboard."""
    st.title('Distribution & Risk')
    st.caption('Daily return distributions, trading activity and downside risk.')

    try:
        returns = load_shared_data()
    except Exception as error:
        st.error(f'Could not load the shared return data: {error}')
        return

    if returns.empty:
        st.warning('No return data is currently available.')
        return

    available_assets = [asset for asset in ASSET_ORDER if asset in returns.columns]
    if not available_assets:
        st.warning('No supported assets were found in the shared data.')
        return

    st.markdown(METHOD_NOTES)

    with st.sidebar:
        st.header('Distribution & Risk')
        selected_assets = st.multiselect('Assets', available_assets, default=available_assets)
        selected_period = st.selectbox('Analysis period', PERIODS, index=3)
        confidence = st.selectbox('VaR confidence level', [0.95, 0.99], index=0, format_func=lambda value: f'{value:.0%}')
        rolling_window = st.slider('Rolling volatility window (trading days)', min_value=10, max_value=120, value=30, step=5)
        kde_bandwidth = st.slider('KDE bandwidth', min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    if not selected_assets:
        st.info('Select at least one asset from the sidebar.')
        return

    selected_returns = returns[selected_assets].copy()
    selected_returns = selected_returns.dropna(how='all')
    period_returns = period_slice(selected_returns, selected_period)

    st.subheader('Risk summary')
    risk_data = risk_table(selected_returns, selected_assets, [selected_period])
    if risk_data.empty:
        st.info('There are not enough observations to calculate the risk statistics.')
    else:
        st.dataframe(risk_data, use_container_width=True, hide_index=True)

    st.subheader('3D joint distribution')
    st.write('The surface shows how daily returns and trading activity combine. Peaks represent combinations that occur more frequently in the sample.')
    try:
        with st.spinner('Loading trading-volume data...'):
            volume_data, volume_sources = load_volume_data(selected_returns.index.min(), selected_returns.index.max())
    except Exception as error:
        st.error(f'Could not load trading-volume data: {error}')
        volume_data, volume_sources = (pd.DataFrame(), {})

    if volume_data.empty:
        st.warning('Trading-volume data is unavailable, so the joint distribution cannot be displayed.')
    else:
        st.caption('Volume source: ' + ' | '.join(f'{asset}: {volume_sources.get(asset, "Unavailable")}' for asset in selected_assets))
        distribution_specs = [(asset, selected_period) for asset in selected_assets]
        joint_fig, contour_fig, joint_error = make_joint_distribution_3d(selected_returns, volume_data, distribution_specs, kde_bandwidth)
        if joint_error:
            st.warning(joint_error)
        else:
            st.plotly_chart(joint_fig, use_container_width=True)
            with st.expander('View 2D density contours'):
                st.plotly_chart(contour_fig, use_container_width=True)

    st.subheader('Rolling volatility')
    st.plotly_chart(make_rolling_volatility(selected_returns, selected_assets, rolling_window), use_container_width=True)

    st.subheader('Trading activity and absolute returns')
    activity_data = activity_relationship_table(selected_returns, volume_data, selected_assets) if not volume_data.empty else pd.DataFrame()
    if activity_data.empty:
        st.info('There is not enough aligned volume data to compare low- and high-activity days.')
    else:
        st.dataframe(activity_data, use_container_width=True, hide_index=True)

    st.subheader('Distribution and variance tests')
    statistical_tests = tests_table(selected_returns, selected_assets)
    if statistical_tests.empty:
        st.info('There are not enough observations on both sides of the 2025 split to run the tests.')
    else:
        st.dataframe(statistical_tests, use_container_width=True, hide_index=True)

    st.subheader('Value at Risk backtesting')
    st.caption(f'Historical {confidence:.0%} VaR breach tests. A breach occurs when the daily return falls below the historical VaR threshold.')
    backtest_data = var_breach_table(selected_returns, selected_assets, confidence)
    if backtest_data.empty:
        st.info('At least 250 observations are needed for the VaR backtest.')
    else:
        st.dataframe(backtest_data, use_container_width=True, hide_index=True)
        st.subheader('Download results')
    if not risk_data.empty:
        csv_data = risk_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Download risk statistics (CSV)',
            data=csv_data,
            file_name='distribution_risk_statistics.csv',
            mime='text/csv'
        )

    with st.expander('Data and methodology details'):
        st.markdown(METHOD_NOTES)
        st.write('Volume data is downloaded from Yahoo Finance and cached locally. The dashboard uses futures volume where coverage is sufficient and ETF proxies otherwise.')
        st.write('The analysis uses daily log returns and trading-volume percentiles. Missing observations are excluded from calculations requiring aligned data.')

    st.caption('Panel 4 – Distribution & Risk | Michael')


def overview():
    """Display a compact preview of Panel 4 on the dashboard overview page."""
    try:
        returns = load_shared_data()
    except Exception:
        st.warning('Distribution & Risk preview is currently unavailable.')
        return

    available_assets = [asset for asset in ASSET_ORDER if asset in returns.columns]
    if not available_assets:
        st.info('No supported assets are available for the Distribution & Risk preview.')
        return

    st.subheader('Distribution & Risk')
    st.caption('A quick look at volatility and downside risk across the selected markets.')

    preview_assets = [asset for asset in ['Gold', 'Silver', 'Oil', 'S&P 500'] if asset in available_assets]
    preview_data = risk_table(returns, preview_assets, ['Since 2025'])

    if preview_data.empty:
        st.info('There are not enough observations to display the risk preview.')
    else:
        st.dataframe(
            preview_data[['Asset', 'Observations', 'Annualised volatility (%)', '95% VaR (%)', 'Maximum drawdown (%)']],
            use_container_width=True,
            hide_index=True
        )

    with st.expander('View distribution methodology'):
        st.markdown(METHOD_NOTES)