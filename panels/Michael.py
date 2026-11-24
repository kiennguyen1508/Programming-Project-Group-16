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