"""
Panel: Regression analysis of silver, oil and gold (Charlie).
"""
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go

from data_loader import get_prices, get_returns, get_macro_changes

SILVER_PEAK = get_prices("2025-01-01")["Silver"].idxmax()        # silver peak

PHASES = {"Before": ("2015-01-01", "2024-12-31"),
          "Boom": ("2025-01-01", SILVER_PEAK),
          "Crash": (SILVER_PEAK + pd.Timedelta(days = 1), None)}

YEARS = {"2015-2024": ("2015-01-01", "2024-12-31"),
         "2025": ("2025-01-01", "2025-12-31"),
         "2026 YTD": ("2026-01-01", None)}

RELATIONSHIPS = {"Silver vs Gold": ("Silver", ["Gold"]),
                 "Silver vs Gold + Copper": ("Silver", ["Gold", "Copper"]),
                 "Oil vs Gold": ("Oil", ["Gold"])}

PERIOD_COLOURS = {"Before": "#7C8798", "Boom": "#B45309", "Crash": "#2563EB",
                  "2015-2024": "#7C8798", "2025": "#B45309", "2026 YTD": "#2563EB"}

METHOD_NOTES = """
**Data.** Daily closing prices from Yahoo Finance: gold (GC=F), silver (SI=F) and Brent crude (BZ=F) futures,
plus COMEX copper futures (HG=F) from the shared macro file. Returns are daily log returns,
ln(P today) - ln(P yesterday), which are the first differences of log prices. Using returns instead of prices
avoids a spurious regression, because prices trend over time and two trending series can look related when they are not.

**Periods.** The boom and crash split is at silver's highest closing price since January 2025, found from the data.
Before is 2015 to 2024, Boom runs from January 2025 to the peak, and Crash is everything after the peak.

**Model.** OLS by matrix formula, β̂ = (X'X)⁻¹X'Y, written by hand with NumPy and checked against statsmodels
in `charlie_regression.ipynb` (the estimates match exactly). In Silver vs Gold + Copper, each β̂ is ceteris paribus:
it shows the effect of one metal holding the other constant.

**Standard errors.** White robust standard errors by default (the HC1 version, with an n / (n - k - 1) correction),
because daily return volatility is not constant over time (heteroskedasticity). The toggle switches to ordinary
standard errors, which assume constant error variance. β̂ is the same either way; only the uncertainty changes.

**Tests.** Two t-tests for each slope: H0: β = 0 (no link) and H0: β = 1 (one-for-one). The confidence interval is
β̂ +/- t_crit x SE, at the level chosen on the slider.

**Limitations.**
- The regression shows association, not causation. It describes how silver moved with gold on the same day;
  it does not forecast future returns or show that gold causes silver to move.
- Daily returns have fat tails, so a few extreme days (such as the large falls after silver's peak) can pull β̂ up.
  The robust standard errors make the intervals wider to reflect this.
- The Boom and Crash periods are much shorter than Before, so their estimates are less precise.
- Brent crude and the metals do not close at exactly the same time each day, which can bias daily betas towards zero.
- Futures prices jump slightly when contracts roll over.
- R² always rises when a regressor is added, so it should not be compared between Silver vs Gold and Silver vs Gold + Copper.
"""

@st.cache_data
def load_returns():
    """
    Load the shared daily returns once, then reuse them on every rerun.
    """
    returns = get_returns()
    copper = get_macro_changes()["Copper"]
    return returns.join(copper)


def run_ols(y, x, alpha_level=0.05, robust=True):
    """
    OLS by matrix formula, with White robust or ordinary standard errors.
    Returns beta_hat, se, t_crit, R squared and n.
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
    if robust:
        meat = (X.T * u_hat ** 2) @ X                                           # inner part of White sandwich estimator
        var_beta_hat = X_prime_X_inv @ meat @ X_prime_X_inv * n / df            # White sandwich estimator
    else:
        sigma_sq_hat = SSR / df
        var_beta_hat = sigma_sq_hat * X_prime_X_inv
    se_beta_hat = np.sqrt(np.diag(var_beta_hat))

    t_crit = stats.t.ppf(1 - alpha_level / 2, df)
    SST = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - SSR / SST
    return beta_hat, se_beta_hat, t_crit, r_squared, n


def results_by_period(returns, y_name, regressors, period_dates, periods, alpha_level=0.05, robust=True):
    """
    Run the regression for each chosen period and return one row per period and regressor.
    """
    rows = []
    for label in period_dates:
        if label not in periods:
            continue
        start, end = period_dates[label]
        data = returns.loc[start:end, [y_name] + regressors].dropna()
        beta_hat, se, t_crit, r2, n = run_ols(data[y_name], data[regressors], alpha_level, robust)

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
        data = returns.loc[start:end, [y_name] + regressors].dropna() * 100         # returns in %
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


def make_3d(returns, y_name, regressors, period_dates, label, alpha_level=0.05):
    """
    3D view of the multiple regression for one period.
    Each dot is one trading day, and the flat surface is the fitted regression plane.
    """
    start, end = period_dates[label]
    data = returns.loc[start:end, [y_name] + regressors].dropna() * 100         # returns in %
    beta_hat, se, t_crit, r2, n = run_ols(data[y_name], data[regressors], alpha_level)
    x1 = regressors[0]
    x2 = regressors[1]

    edges_1 = np.array([data[x1].min(), data[x1].max()])
    edges_2 = np.array([data[x2].min(), data[x2].max()])
    grid_1, grid_2 = np.meshgrid(edges_1, edges_2)                  # the 4 corners of the plane
    plane = beta_hat[0] + beta_hat[1] * grid_1 + beta_hat[2] * grid_2

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x = data[x1],
        y = data[x2],
        z = data[y_name],
        mode = "markers",
        name = "Daily returns",
        marker = dict(size = 2, opacity = 0.5, color = PERIOD_COLOURS[label]),
    ))
    fig.add_trace(go.Surface(
        x = grid_1,
        y = grid_2,
        z = plane,
        name = "Fitted plane",
        opacity = 0.5,
        showscale = False,
        colorscale = [[0, "#E0B252"], [1, "#E0B252"]],
    ))
    fig.update_layout(
        height = 550,
        title = f"{label}: {y_name} = {beta_hat[0]:.2f} + {beta_hat[1]:.2f} x {x1} + {beta_hat[2]:.2f} x {x2}",
        scene = dict(
            xaxis_title = f"{x1} (%)",
            yaxis_title = f"{x2} (%)",
            zaxis_title = f"{y_name} (%)",
            aspectmode = "manual",
            aspectratio = dict(x = 1.5, y = 1.5, z = 1),
            camera = dict(eye = dict(x = 1, y = 2.1, z = 0.6),
                          center = dict(x = 0, y = 0, z = -0.25)),
        ),
        margin = dict(l=0, r=0, t=40, b=0),
    )
    return fig


def describe(row, y_name, regressors, confidence):
    """
    One plain-English sentence for one beta, using its confidence interval.
    """
    name = row["Regressor"]
    held = ""
    for other in regressors:
        if other != name:
            held = f", holding {other.lower()} constant"

    if row["CI low"] > 1:
        verdict = "more than one-for-one"
    elif row["CI high"] < 0:
        verdict = "a negative link"
    elif row["CI low"] > 0 and row["CI high"] < 1:
        verdict = "less than one-for-one"
    elif row["CI low"] > 0:
        verdict = "consistent with one-for-one"
    else:
        verdict = "not clearly different from zero"

    return (f"**{row['Period']}:** a 1% move in {name.lower()}{held} was associated with about a "
            f"{row['Beta']:.2f}% move in {y_name.lower()} on average "
            f"({confidence:.0%} CI {row['CI low']:.2f} to {row['CI high']:.2f}). This is {verdict}.")


def render():
    """
    Draw my panel inside the dashboard.
    """
    st.subheader("Regression analysis: silver, oil and gold")
    st.caption("By Charlie O'Sullivan | Data: Yahoo Finance daily prices")

    returns = load_returns()

    relationship = st.radio(
        "Relationship",
        options = list(RELATIONSHIPS),
        horizontal = True,
        key = "cha_rel",
    )
    y_name, regressors = RELATIONSHIPS[relationship]

    col_split, col_conf, col_robust = st.columns([2, 1, 1], vertical_alignment = "bottom")
    with col_split:
        scheme = st.radio(
            "Split the data by",
            options = ["Boom and crash", "Calendar years"],
            horizontal = True,
            key = "cha_scheme",
        )
    with col_conf:
        confidence = st.select_slider(
            "Confidence level",
            options = [0.90, 0.95, 0.99],
            value = 0.95,
            format_func = lambda c: f"{c:.0%}",
            key = "cha_conf",
        )
    with col_robust:
        robust = st.toggle(
            "Robust standard errors (White)",
            value = True,
            key = "cha_robust",
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

    if not chosen:
        st.warning("Pick at least one period to see the results.")
        return

    by_period = results_by_period(returns, y_name, regressors, period_dates, chosen, 1 - confidence, robust)

    views = ["Scatter", "Dot and whisker"]
    if len(regressors) == 2:
        views.append("3D plane")

    view = st.radio(
        "Chart",
        options = views,
        horizontal = True,
        key = f"cha_view_{len(regressors)}",
    )
    if view == "Scatter":
        st.plotly_chart(make_scatter(returns, y_name, regressors, period_dates, chosen, 1 - confidence))
    elif view == "Dot and whisker":
        st.plotly_chart(make_dot_whisker(by_period, regressors, period_dates, confidence))
    else:
        which = st.selectbox(
            "Period for the 3D view",
            options = chosen,
            key = "cha_3d",
        )
        st.plotly_chart(make_3d(returns, y_name, regressors, period_dates, which, 1 - confidence))
        st.caption("Drag to rotate. Each dot is one trading day, and the gold surface is the fitted regression plane.")

    st.markdown("**Regression results**")
    if robust:
        st.caption(f"Confidence intervals at the {confidence:.0%} level, using White robust standard errors")
    else:
        st.caption(f"Confidence intervals at the {confidence:.0%} level, assuming constant error variance")
    st.dataframe(by_period.round(3), hide_index = True, width = "stretch")

    with st.expander("What the results mean", expanded = True):
        for i in range(len(by_period)):
            st.markdown(describe(by_period.iloc[i], y_name, regressors, confidence))

    with st.expander("Method and limitations"):
        st.markdown(METHOD_NOTES)


def overview():
    """
    Small preview for the overview grid: silver's beta on gold before, during and after the boom.
    """
    returns = load_returns()
    by_period = results_by_period(returns, "Silver", ["Gold"], PHASES, list(PHASES))
    crash_beta = by_period[by_period["Period"] == "Crash"]["Beta"].iloc[0]

    fig = make_dot_whisker(by_period, ["Gold"], PHASES, 0.95)
    fig.update_layout(height = 300, xaxis_title = None, margin = dict(l=10, r=10, t=30, b=0))
    fig.update_yaxes(showticklabels = False)
    st.plotly_chart(fig, config = {"staticPlot": True}, key = "cha_overview_chart")
    st.caption(f"After silver's peak on {SILVER_PEAK:%d %b %Y}, a 1% move in gold was associated "
               f"with about a {crash_beta:.1f}% move in silver")