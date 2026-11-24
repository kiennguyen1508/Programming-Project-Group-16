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

RELATIONSHIP_NOTES = {"Silver vs Gold": "How did silver react to gold's movements?",
                      "Silver vs Gold + Copper": "Is silver a precious metal like gold, or an industrial metal like copper?",
                      "Oil vs Gold": "Did oil and gold move together, and did that change in 2026, the year of the Iran tension?"}

BETA_NOTES = {"Silver vs Gold": "β is silver's average daily move for each 1% daily move in gold. β = 1 means one-for-one, "
                                "above 1 means silver's moves were bigger than gold's and below 1 means smaller. If the "
                                "confidence interval includes 1, silver was not clearly different from one-for-one.",
              "Silver vs Gold + Copper": "There is one β for each metal: silver's average daily move for a 1% daily move "
                                         "in that metal, holding the other metal constant. The bigger β shows which "
                                         "metal silver tracked more closely.",
              "Oil vs Gold": "β is oil's average daily move for each 1% daily move in gold. A positive β means they moved "
                             "in the same direction, a negative β in opposite directions. If the confidence interval "
                             "includes 0, there is no clear link."}

CAVEATS = {"Silver vs Gold": "These are same-day associations. They show how much silver amplified gold's moves, "
                               "not that gold caused them, and they do not forecast future returns.",
           "Silver vs Gold + Copper": "Comparing the two βs suggests whether silver behaved more like a precious or an "
                                        "industrial metal, but it is not a formal test. Copper is also more volatile "
                                        "than gold, which makes its β look smaller.",
           "Oil vs Gold": "A link found in one period should not be assumed to hold in another. These are same-day "
                            "associations, not causes, and they do not forecast future returns."}

ONE_FOR_ONE = {"Silver vs Gold": True,
               "Silver vs Gold + Copper": True,
               "Oil vs Gold": False}          # one-for-one doesnt make sense for oil

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
in `charlie_regression.ipynb` (the estimates and robust standard errors match exactly). In Silver vs Gold + Copper,
each β̂ is ceteris paribus: it shows the effect of one metal holding the other constant.

**Standard errors.** White robust standard errors by default (the HC1 version, with an n / (n - k - 1) correction),
because daily return volatility is not constant over time (heteroskedasticity). The toggle switches to ordinary
standard errors, which assume constant error variance. β̂ is the same either way; only the uncertainty changes.

**Tests.** Two t-tests for each slope: H0: β = 0 (no link) and, for silver only, H0: β = 1 (one-for-one). The confidence interval is
β̂ +/- t_crit x SE, at the level chosen on the slider.
A change in β between two periods is tested with z = (β̂₂ - β̂₁) / √(SE₁² + SE₂²), treating the periods as independent samples.

**Limitations.**
- The regression shows association, not causation. It describes how silver moved with gold on the same day;
  it does not forecast future returns or show that gold causes silver to move.
- Daily returns have fat tails, so a few extreme days (such as the large falls after silver's peak) can have a big
  influence on β̂. Robust standard errors allow for changing volatility, but they do not remove the effect of these days.
- The standard errors assume each day's error is unrelated to the next day's. Robust standard errors do not correct
  for autocorrelation, although daily returns usually show little of it.
- The Boom and Crash periods are much shorter than Before, so their estimates are less precise.
- Brent crude and the metals do not close at exactly the same time each day, which can bias daily betas towards zero.
- Futures prices jump slightly when contracts roll over.
- R² always rises when a regressor is added, so it should not be compared between Silver vs Gold and Silver vs Gold + Copper.
- Copper is more volatile than gold, so comparing silver's β on each is a rough guide to precious versus industrial, not a formal test.
"""

@st.cache_data
def load_returns():
    """
    Load the shared daily returns once, then reuse them on every rerun.
    """
    returns = get_returns()
    copper = get_macro_changes()["Copper"]
    return returns.join(copper)


@st.cache_data
def silver_story():
    """
    How far silver rose from the start of 2025 to its peak, and how far it has fallen since.
    """
    silver = get_prices("2025-01-01")["Silver"]
    rise = silver.loc[SILVER_PEAK] / silver.iloc[0] - 1
    fall = silver.iloc[-1] / silver.loc[SILVER_PEAK] - 1
    return rise, fall


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


def results_by_period(returns, y_name, regressors, period_dates, periods, alpha_level=0.05, robust=True, one_for_one=True):
    """
    Run the regression for each chosen period and return one row per period and regressor.
    """
    rows = []
    for label in period_dates:
        if label not in periods:
            continue
        start, end = period_dates[label]
        data = returns.loc[start:end, [y_name] + regressors].dropna()
        if len(data) <= len(regressors) + 1:          # if not enough data points to run the regression
            continue
        beta_hat, se, t_crit, r2, n = run_ols(data[y_name], data[regressors], alpha_level, robust)

        for i in range(len(regressors)):
            name = regressors[i]
            b = beta_hat[i + 1]         # +1 skips the intercept
            s = se[i + 1]               # +1 skips the intercept
            row = {
                "Period": label,
                "Regressor": name,
                "Beta": b,
                "Std. Error": s,
                "CI low": b - t_crit * s,
                "CI high": b + t_crit * s,
                "t (β=0)": b / s,
            }
            if one_for_one:
                row["t (β=1)"] = (b - 1) / s
            row["R²"] = r2
            row["n"] = n
            rows.append(row)

    return pd.DataFrame(rows)


def make_scatter(returns, y_name, regressors, period_dates, periods):
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
        beta_hat, se, t_crit, r2, n = run_ols(data[y_name], data[regressors])

        fig.add_trace(go.Scatter(
            x = data["Gold"],
            y = data[y_name],
            mode = "markers",
            name = label,
            legendgroup = label,
            showlegend = False,
            marker = dict(size = 4, opacity = 0.7, color = PERIOD_COLOURS[label]),
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
        margin = dict(l = 10, r = 10, t = 30, b = 10),
    )
    return fig


def make_dot_whisker(by_period, regressors, period_dates, confidence, one_for_one=True):
    """
    Each beta as a dot with its confidence interval, one colour per period.
    Dashed lines at 0 (no link) and, for silver, at 1 (one-for-one with gold).
    """
    fig = go.Figure()
    for label in period_dates:
        rows = by_period[by_period["Period"] == label]
        if rows.empty:
            continue

        fig.add_trace(go.Scatter(
            x = rows["Beta"],
            y = rows["Regressor"],
            mode = "markers+text",
            text = rows["Beta"].round(2),
            textposition = "top center",
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
    if one_for_one:
        fig.add_vline(x = 1, line_dash = "dash", line_color = "grey",
                      annotation_text = "β = 1", annotation_position = "top")
    fig.update_layout(
        height = 150 + 80 * len(regressors),
        xaxis_title = f"Beta with {confidence:.0%} confidence interval",
        scattermode = "group",
        margin = dict(l = 80, r = 10, t = 40, b = 50),
    )
    return fig


def make_3d(returns, y_name, regressors, period_dates, label):
    """
    3D view of the multiple regression for one period.
    Each dot is one trading day, and the flat surface is the fitted regression plane.
    """
    start, end = period_dates[label]
    data = returns.loc[start:end, [y_name] + regressors].dropna() * 100         # returns in %
    beta_hat, se, t_crit, r2, n = run_ols(data[y_name], data[regressors])
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
        marker = dict(size = 2, opacity = 0.7, color = PERIOD_COLOURS[label]),
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
        margin = dict(l = 0, r = 0, t = 40, b = 0),
    )
    return fig


def describe(row, y_name, regressors, confidence, one_for_one=True):
    """
    One plain-English sentence for one beta, using its confidence interval.
    """
    name = row["Regressor"]
    held = ""
    for other in regressors:
        if other != name:
            held = f", holding {other.lower()} constant,"

    if row["Beta"] < 0:
        direction = "fall"
    else:
        direction = "rise"

    if not one_for_one:
        if row["CI low"] > 0:
            verdict = "a positive link"
        elif row["CI high"] < 0:
            verdict = "a negative link"
        else:
            verdict = "not clearly different from zero"
    elif row["CI low"] > 1:
        verdict = "more than one-for-one"
    elif row["CI high"] < 0:
        verdict = "a negative link"
    elif row["CI low"] > 0 and row["CI high"] < 1:
        verdict = "less than one-for-one"
    elif row["CI low"] > 0:
        verdict = "consistent with one-for-one"
    else:
        verdict = "not clearly different from zero"

    return (f"**{row['Period']}:** a 1% daily rise in {name.lower()}{held} was associated with about a "
            f"{abs(row['Beta']):.2f}% daily {direction} in {y_name.lower()} on average "
            f"({confidence:.0%} CI {row['CI low']:.2f} to {row['CI high']:.2f}). This is {verdict}.")


def compare(by_period, y_name, name, confidence, one_for_one=True, significant_only=False):
    """
    z-test for whether beta changed from one period to the next.
    Set significant_only to only keep the changes that are significant.
    """
    rows = by_period[by_period["Regressor"] == name]
    y = y_name.lower()
    x = name.lower()
    alpha_level = 1 - confidence
    z_crit = stats.norm.ppf(1 - alpha_level / 2)
    lines = []
    for i in range(1, len(rows)):
        earlier = rows.iloc[i - 1]
        later = rows.iloc[i]
        z = (later["Beta"] - earlier["Beta"]) / np.sqrt(earlier["Std. Error"] ** 2 + later["Std. Error"] ** 2)
        if significant_only and abs(z) <= z_crit:
            continue

        if z > z_crit and one_for_one:
            result = f"a significant rise at the {confidence:.0%} level, so {y} reacted more to each move in {x}"
        elif z > z_crit:
            result = f"a significant rise at the {confidence:.0%} level, so {y} moved more with {x}, or less against it"
        elif z < -z_crit and one_for_one:
            result = f"a significant fall at the {confidence:.0%} level, so {y} reacted less to each move in {x}"
        elif z < -z_crit:
            result = f"a significant fall at the {confidence:.0%} level, so {y} moved less with {x}, or more against it"
        else:
            result = "not a significant change"

        lines.append(f"**{earlier['Period']} to {later['Period']}:** {y}'s β on {x} went from "
                     f"{earlier['Beta']:.2f} to {later['Beta']:.2f} (z = {z:.2f}), {result}.")
    return lines


def gold_copper_table(by_period):
    """
    Small table of silver's beta on gold and copper for each period within what results mean, so it's easier to read.
    """
    rows = []
    for period in by_period["Period"].unique():
        in_period = by_period[by_period["Period"] == period]
        gold = in_period[in_period["Regressor"] == "Gold"].iloc[0]
        copper = in_period[in_period["Regressor"] == "Copper"].iloc[0]
        if gold["Beta"] > copper["Beta"]:
            closer = "Gold"
        else:
            closer = "Copper"
        rows.append({"Period": period,
                     "β on gold": round(gold["Beta"], 2),
                     "β on copper": round(copper["Beta"], 2),
                     "Silver moved more with": closer})
    return pd.DataFrame(rows)


def takeaway(by_period, relationship):
    """
    What the results mean in practice, shown in the green box.
    """
    gold = by_period[by_period["Regressor"] == "Gold"]
    latest = gold.iloc[-1]

    if relationship == "Silver vs Gold":
        above = (gold["CI low"] > 1).sum()
        if above == len(gold):
            text = ("Silver acted like a geared version of gold: when gold moved, silver tended to move further in "
                    "the same direction. For someone holding silver, that means bigger gains when gold rises and "
                    "bigger losses when it falls.")
        elif above > 0:
            text = ("In some periods silver moved further than gold, so holding silver was riskier than gold at "
                    "times, but not consistently.")
        else:
            text = "Silver did not clearly move more than gold here, so holding silver was not clearly riskier than gold."

    elif relationship == "Silver vs Gold + Copper":
        copper = by_period[by_period["Regressor"] == "Copper"]
        gold_bigger = (gold["Beta"].to_numpy() > copper["Beta"].to_numpy()).sum()
        if gold_bigger == len(gold):
            text = ("Even with industrial demand behind part of the boom, silver still tracked gold more than copper, "
                    "so it traded more like a precious metal than an industrial one.")
        elif gold_bigger > 0:
            text = "Silver switched between tracking gold and copper, so whether it trades as precious or industrial depends on the period."
        else:
            text = "Silver tracked copper more than gold, so it traded more like an industrial metal than a precious one."

    else:
        if latest["CI high"] < 0:
            text = (f"In {latest['Period']}, days when gold rose tended to be days when oil fell. Holding both would "
                    "have offset some risk, because one tended to rise when the other fell.")
        elif latest["CI low"] > 0:
            text = (f"In {latest['Period']}, oil and gold tended to rise and fall together, so holding both gave "
                    "little protection against a fall in either.")
        else:
            text = f"In {latest['Period']}, gold's daily move told us little about oil's, so they moved largely independently."

        if latest["R²"] < 0.05 and (latest["CI high"] < 0 or latest["CI low"] > 0):
            text = text + (f" The link was weak, though: gold's moves explained only about {latest['R²']:.0%} of oil's "
                           "daily moves.")
    return text


def answer(by_period, relationship, confidence):
    """
    Plain-English answer to the question behind the chosen relationship.
    """
    gold = by_period[by_period["Regressor"] == "Gold"]
    n_periods = len(gold)
    latest = gold.iloc[-1]
    if n_periods > 1:
        every = " in every period"
    else:
        every = ""

    if relationship == "Silver vs Gold":
        above = (gold["CI low"] > 1).sum()
        if above == n_periods:
            headline = f"More strongly than gold{every}."
        elif above > 0:
            headline = "More strongly than gold in some periods."
        else:
            headline = "Not clearly more strongly than gold."
        if n_periods > 1:
            evidence = f"Silver's moves were significantly bigger than gold's in {above} of {n_periods} periods"
        elif above == 1:
            evidence = "Silver's moves were significantly bigger than gold's"
        else:
            evidence = "Silver's moves were not significantly bigger than gold's"
        text = (f"**How did silver react to gold's movements?** {headline}\n\n"
                f"In the {latest['Period']} period, a 1% daily move in gold was associated with about a "
                f"{latest['Beta']:.1f}% daily move in silver on average. {evidence}, at the {confidence:.0%} "
                f"confidence level.")

    elif relationship == "Silver vs Gold + Copper":
        copper = by_period[by_period["Regressor"] == "Copper"]
        latest_copper = copper.iloc[-1]
        gold_bigger = (gold["Beta"].to_numpy() > copper["Beta"].to_numpy()).sum()
        if gold_bigger == n_periods:
            headline = f"Stronger estimated link with gold{every}, so silver behaved more like a precious metal."
        elif gold_bigger > 0:
            headline = ("Mixed: stronger with gold in some periods, copper in others, so silver behaved partly "
                        "like a precious metal and partly like an industrial one.")
        else:
            headline = f"Stronger estimated link with copper{every}, so silver behaved more like an industrial metal."
        if n_periods > 1:
            evidence = f" Silver moved more with gold than copper in {gold_bigger} of {n_periods} periods."
        else:
            evidence = ""
        text = (f"**Is silver a precious or an industrial metal?** {headline}\n\n"
                f"In the {latest['Period']} period, holding the other metal constant, a 1% daily move in gold was associated "
                f"with about a {latest['Beta']:.1f}% daily move in silver, and a 1% daily move in copper with about a "
                f"{latest_copper['Beta']:.1f}% daily move.{evidence}")

    else:
        positive = (gold["CI low"] > 0).sum()
        negative = (gold["CI high"] < 0).sum()
        clear = positive + negative
        if clear == 0:
            headline = "No clear link."
        elif positive > 0 and negative > 0:
            headline = "The link changed direction."
        elif clear < n_periods:
            headline = "Only sometimes."
        elif negative == n_periods:
            headline = "Yes, but in opposite directions."
        else:
            headline = "Yes, in the same direction."

        if latest["Beta"] < 0:
            move = "fall"
        else:
            move = "rise"
        if n_periods > 1:
            evidence = f"The link was significantly different from zero in {clear} of {n_periods} periods"
        elif clear == 1:
            evidence = "The link was significantly different from zero"
        else:
            evidence = "The link was not significantly different from zero"
        text = (f"**Did oil and gold move together?** {headline}\n\n"
                f"In the {latest['Period']} period, a 1% daily rise in gold was associated with about a "
                f"{abs(latest['Beta']):.1f}% daily {move} in oil on average. {evidence}, at the {confidence:.0%} "
                f"confidence level.")
    return text


def render():
    """
    Draw my panel inside the dashboard.
    """
    st.subheader("Regression analysis: silver, oil and gold")
    st.caption("By Charlie O'Sullivan | Data: Yahoo Finance daily prices")
    rise, fall = silver_story()
    st.caption(f"Silver rose {rise:.0%} from the start of 2025 to its peak on {SILVER_PEAK:%d %b %Y}, "
               f"then fell {abs(fall):.0%}.")

    returns = load_returns()

    relationship = st.segmented_control(
        "Relationship",
        options = list(RELATIONSHIPS),
        default = "Silver vs Gold",
        key = "cha_rel",
    )
    if relationship is None:
        relationship = "Silver vs Gold"
    y_name, regressors = RELATIONSHIPS[relationship]
    st.caption(RELATIONSHIP_NOTES[relationship])

    col_split, col_conf, col_robust = st.columns([2, 1, 1], vertical_alignment = "bottom")
    with col_split:
        scheme = st.segmented_control(
            "Split the data by",
            options = ["Boom and crash", "Calendar years"],
            default = "Boom and crash",
            key = "cha_scheme",
        )
        if scheme is None:
            scheme = "Boom and crash"
    with col_conf:
        confidence = st.select_slider(
            "Confidence level",
            options = [0.90, 0.95, 0.99],
            value = 0.95,
            format_func = lambda c: f"{c:.0%}",
            help = "How strict the test is before calling a result significant. Higher means a wider confidence interval.",
            key = "cha_conf",
        )
    with col_robust:
        robust = st.toggle(
            "Robust standard errors (White)",
            value = True,
            help = "Allows for some days being much more volatile than others. Only the standard errors and confidence intervals change; β does not change.",
            key = "cha_robust",
        )

    if scheme == "Boom and crash":
        period_dates = PHASES
    else:
        period_dates = YEARS

    chosen = st.pills(
        "Periods",
        options = list(period_dates),
        selection_mode = "multi",
        default = list(period_dates),
        key = f"cha_per_{scheme}",
    )
    if scheme == "Boom and crash":
        st.caption(f"Before: 2015-2024 | Boom: Jan 2025 to silver's peak on {SILVER_PEAK:%d %b %Y} | Crash: after the peak")
        if y_name == "Oil":
            st.caption("These periods are set by silver's peak, not by oil. Switch to calendar years to see 2026 on its own.")

    if not chosen:
        st.warning("Pick at least one period to see the results.")
        return

    one_for_one = ONE_FOR_ONE[relationship]
    by_period = results_by_period(returns, y_name, regressors, period_dates, chosen, 1 - confidence, robust,
                                  one_for_one)

    if by_period.empty:                 # in case data for every chosen period is missing
        st.warning("None of the chosen periods has enough trading days to run the regression.")
        return
    ran = list(by_period["Period"].unique())        # charts only use periods that have results

    views = ["Scatter", "Dot and whisker"]
    if len(regressors) == 2:
        views.insert(0, "3D plane")

    view = st.segmented_control(
        "Chart",
        options = views,
        default = views[0],
        key = f"cha_view_{len(regressors)}",
    )
    if view is None:
        view = views[0]
    if view == "Scatter":
        st.plotly_chart(make_scatter(returns, y_name, regressors, period_dates, ran))
        if len(regressors) == 2:
            st.caption("Each dot is one trading day. The lines show silver's move for each 1% daily move in gold, with copper held at its average.")
    elif view == "Dot and whisker":
        st.plotly_chart(make_dot_whisker(by_period, regressors, period_dates, confidence, one_for_one))
        note = "Each dot is β and the line through it is the confidence interval. If the line crosses 0, there is no clear link."
        if one_for_one:
            note = note + " If it crosses 1, silver's moves were not clearly different from gold's."
        st.caption(note)
    else:
        which = st.selectbox(
            "Period for the 3D view",
            options = ran,
            key = f"cha_3d_{'_'.join(ran)}",
        )
        st.plotly_chart(make_3d(returns, y_name, regressors, period_dates, which))
        st.caption("Drag to rotate. Each dot is one trading day, and the gold surface is the fitted regression plane.")

    st.info(answer(by_period, relationship, confidence))

    col_title, col_csv = st.columns([4, 1], vertical_alignment = "bottom")
    with col_title:
        st.markdown("**Regression results**")
        if robust:
            st.caption(f"Confidence intervals at the {confidence:.0%} level, using White robust standard errors")
        else:
            st.caption(f"Confidence intervals at the {confidence:.0%} level, assuming constant error variance")
    with col_csv:
        st.download_button(
            "Download CSV",
            data = by_period.to_csv(index = False).encode("utf-8"),
            file_name = f"regression_{relationship.replace(' ', '_').lower()}.csv",
            mime = "text/csv",
            key = "cha_download",
        )
    st.dataframe(by_period.round(3), hide_index = True, width = "stretch")
    st.caption(f"Beta is {y_name.lower()}'s average daily move for each 1% daily move in the regressor. CI low to CI high "
               "is the confidence interval, a range of plausible values for β. n is the number of trading days.")

    with st.expander("What the results mean", expanded = True):
        st.markdown("##### What β tells us")
        st.markdown(BETA_NOTES[relationship])

        st.markdown("##### Results by period")
        if len(regressors) == 1:
            lines = []
            for i in range(len(by_period)):
                lines.append("- " + describe(by_period.iloc[i], y_name, regressors, confidence, one_for_one))
            st.markdown("\n".join(lines))
        else:
            st.dataframe(gold_copper_table(by_period), hide_index = True)

        st.markdown("##### What changed between periods")
        changes = []
        for name in regressors:
            changes = changes + compare(by_period, y_name, name, confidence, one_for_one, significant_only = True)
        if len(ran) < 2:
            st.markdown("Pick more than one period to test whether β changed.")
        elif changes:
            st.markdown("\n".join("- " + line for line in changes))
        else:
            st.markdown(f"No β changed significantly between periods at the {confidence:.0%} level, "
                        "so the differences could be down to chance.")

        boom_and_crash = ("Boom" in ran and "Crash" in ran) or ("2025" in ran and "2026 YTD" in ran)
        if relationship == "Silver vs Gold" and boom_and_crash:
            st.caption("Why might β dip in the boom and jump after the peak? β measures how far silver moved with gold "
                       "on the same day, not how much it rose overall. Some of silver's rise may have come from its own "
                       "drivers, such as industrial demand and tight supply, which would weaken its daily link to gold. "
                       "After the peak, both sold off together and silver fell further. These are likely reasons, not "
                       "something the regression tests.")

        st.markdown("##### What to take away")
        st.success(takeaway(by_period, relationship))
        st.caption(CAVEATS[relationship])
        if scheme == "Calendar years":
            st.caption(f"Calendar years cut at 31 Dec 2025, so the weeks just before silver's peak on "
                       f"{SILVER_PEAK:%d %b %Y} fall in 2026 YTD instead of 2025. That is why these results can "
                       "differ from the boom and crash split.")

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
    fig.update_layout(height = 300, xaxis_title = None, margin = dict(l = 10, r = 10, t = 30, b = 0))
    fig.update_yaxes(showticklabels = False)
    st.plotly_chart(fig, config = {"staticPlot": True}, key = "cha_overview_chart")
    st.caption(f"After silver's peak on {SILVER_PEAK:%d %b %Y}, a 1% daily move in gold was associated "
               f"with about a {crash_beta:.1f}% daily move in silver.")