"""
Panel: Silver vs gold regression (Charlie).
"""
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go

from data_loader import get_prices, get_returns

SILVER_PEAK = get_prices("2025-01-01")["Silver"].idxmax()        # silver peak

PHASES = {"Before": ("2015-01-01", "2024-12-31"),
          "Boom": ("2025-01-01", SILVER_PEAK),
          "Crash": (SILVER_PEAK + pd.Timedelta(days = 1), None)}

YEARS = {"2015-2024": ("2015-01-01", "2024-12-31"),
         "2025": ("2025-01-01", "2025-12-31"),
         "2026 YTD": ("2026-01-01", None)}

RELATIONSHIPS = {"Silver vs Gold": ("Silver", ["Gold"]),
                 "Silver vs Gold + S&P 500": ("Silver", ["Gold", "S&P 500"]),
                 "Oil vs Gold": ("Oil", ["Gold"])}

PERIOD_COLOURS = {"Before": "#7C8798", "Boom": "#B45309", "Crash": "#2563EB",
                  "2015-2024": "#7C8798", "2025": "#B45309", "2026 YTD": "#2563EB"}

@st.cache_data
def load_returns():
    """
    Load the shared daily returns once, then reuse them on every rerun.
    """
    return get_returns()


def run_ols(y, x, alpha_level=0.05):
    """
    OLS by matrix formula. Returns beta_hat, se, t_crit, R squared and n.
    """
    n = len(y)
    Y = y.to_numpy()
    X = np.column_stack([np.ones(n), x])
    k = x.shape[1]
    df = n - k - 1

    X_prime_X_inv = np.linalg.inv(X.T @ X)
    beta_hat = X_prime_X_inv @ (X.T @ Y)

    u_hat = Y - X @ beta_hat
    SSR = u_hat @ u_hat
    sigma_sq_hat = SSR / df
    se_beta_hat = np.sqrt(np.diag(sigma_sq_hat * X_prime_X_inv))

    t_crit = stats.t.ppf(1 - alpha_level / 2, df)
    SST = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - SSR / SST
    return beta_hat, se_beta_hat, t_crit, r_squared, n


def results_by_period(returns, y_name, regressors, period_dates, periods, alpha_level=0.05):
    """
    Run the regression for each chosen period and return one row per period and regressor.
    """
    rows = []
    for label in period_dates:
        if label not in periods:
            continue
        start, end = period_dates[label]
        data = returns.loc[start:end]
        beta_hat, se, t_crit, r2, n = run_ols(data[y_name], data[regressors], alpha_level)

        for i in range(len(regressors)):
            name = regressors[i]
            b = beta_hat[i + 1]         # +1 skips the intercept
            s = se[i + 1]               # +1 skips the intercept
            rows.append({
                "Period": label,
                "Regressor": name,
                "Beta": b,
                "Std. Error": s,
                "CI low": b - t_crit * s,
                "CI high": b + t_crit * s,
                "t (β=0)": b / s,
                "t (β=1)": (b - 1) / s,
                "R²": r2,
                "n": n,
            })
    return pd.DataFrame(rows)


def make_scatter(returns, y_name, regressors, period_dates, periods, alpha_level=0.05):
    """
    Scatter of daily returns against gold, with a fitted line for each chosen period.
    Any other regressors are held at their average for that period.
    """
    fig = go.Figure()
    for label in period_dates:
        if label not in periods:
            continue
        start, end = period_dates[label]
        data = returns.loc[start:end] * 100         # returns in %
        beta_hat, se, t_crit, r2, n = run_ols(data[y_name], data[regressors], alpha_level)

        fig.add_trace(go.Scatter(
            x = data["Gold"],
            y = data[y_name],
            mode = "markers",
            name = label,
            legendgroup = label,
            showlegend = False,
            marker = dict(size = 4, opacity = 0.3, color = PERIOD_COLOURS[label]),
        ))

        line_x = np.array([data["Gold"].min(), data["Gold"].max()])
        line_y = beta_hat[0] + beta_hat[1] * line_x
        for i in range(1, len(regressors)):         # hold other regressors constant at their average
            line_y = line_y + beta_hat[i + 1] * data[regressors[i]].mean()

        fig.add_trace(go.Scatter(
            x = line_x,
            y = line_y,
            mode = "lines",
            name = f"{label} (β = {beta_hat[1]:.2f})",
            legendgroup = label,
            line = dict(color = PERIOD_COLOURS[label], width = 3),
        ))

    fig.update_layout(
        height = 420,
        xaxis_title = "Gold daily return (%)",
        yaxis_title = f"{y_name} daily return (%)",
        legend = dict(orientation = "h", y = 1.15),
        margin = dict(l=10, r=10, t=30, b=10),
    )
    return fig


def make_dot_whisker(by_period, regressors, period_dates, confidence):
    """
    Each beta as a dot with its confidence interval, one colour per period.
    Dashed lines at 0 (no link) and 1 (one-for-one with gold).
    """
    fig = go.Figure()
    for label in period_dates:
        rows = by_period[by_period["Period"] == label]
        if rows.empty:
            continue

        fig.add_trace(go.Scatter(
            x = rows["Beta"],
            y = rows["Regressor"],
            mode = "markers",
            orientation = "h",
            name = label,
            marker = dict(size = 12, color = PERIOD_COLOURS[label]),
            error_x = dict(
                type = "data",
                symmetric = False,
                array = rows["CI high"] - rows["Beta"],
                arrayminus = rows["Beta"] - rows["CI low"],
                thickness = 2,
                width = 6,
            ),
        ))

    fig.add_vline(x = 0, line_dash = "dot", line_color = "grey",
                  annotation_text = "β = 0", annotation_position = "top")
    fig.add_vline(x = 1, line_dash = "dash", line_color = "grey",
                  annotation_text = "β = 1", annotation_position = "top")
    fig.update_layout(
        height = 150 + 80 * len(regressors),
        xaxis_title = f"Beta with {confidence:.0%} confidence interval",
        scattermode = "group",
        margin = dict(l=80, r=10, t=40, b=50),
    )
    return fig


def render():
    """
    Draw my panel inside the dashboard.
    """
    st.subheader("Silver vs gold regression")
    st.caption("By Charlie O'Sullivan | Data: Yahoo Finance daily prices")

    returns = load_returns()

    relationship = st.radio(
        "Relationship",
        options = list(RELATIONSHIPS),
        horizontal = True,
        key = "cha_rel",
    )
    y_name, regressors = RELATIONSHIPS[relationship]

    scheme = st.radio(
        "Split the data by",
        options = ["Boom and crash", "Calendar years"],
        horizontal = True,
        key = "cha_scheme",
    )
    if scheme == "Boom and crash":
        period_dates = PHASES
    else:
        period_dates = YEARS

    chosen = st.multiselect(
        "Periods",
        options = list(period_dates),
        default = list(period_dates),
        key = f"cha_per_{scheme}",
    )
    if scheme == "Boom and crash":
        st.caption(f"Before: 2015-2024 | Boom: Jan 2025 to silver's peak on {SILVER_PEAK:%d %b %Y} | Crash: after the peak")

    confidence = st.select_slider(
        "Confidence level",
        options = [0.90, 0.95, 0.99],
        value = 0.95,
        format_func = lambda c: f"{c:.0%}",
        key = "cha_conf",
    )

    if not chosen:
        st.warning("Pick at least one period to see the results.")
        return

    by_period = results_by_period(returns, y_name, regressors, period_dates, chosen, 1 - confidence)

    view = st.radio(
        "Chart",
        options = ["Scatter", "Dot and whisker"],
        horizontal = True,
        key = "cha_view",
    )
    if view == "Scatter":
        st.plotly_chart(make_scatter(returns, y_name, regressors, period_dates, chosen, 1 - confidence))
    else:
        st.plotly_chart(make_dot_whisker(by_period, regressors, period_dates, confidence))

    st.caption(f"Confidence intervals at the {confidence:.0%} level")
    st.dataframe(by_period.round(3), hide_index = True, width = "stretch")