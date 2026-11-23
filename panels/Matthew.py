"""Panel 3: Correlation Heatmap - Matthew"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import get_returns, SPLIT    



NAMELAYOUT = {"S&P 500": "S&P<br>500", "Real Yield": "Real<br>Yield",
"Commodities": "Commod-<br>ities"}
GOLD_BLUE = [[0, "#2E6FD8"], [0.5, "#1C2230"], [1, "#E0A82E"]]
NS_GREY = "#161B26"
LIGHT_TEXT = "#E6E9EF"
SOFT_TEXT = "#9AA4B8"
GOLD, BLUE = "#E0A82E", "#2E6FD8"
EVENTS = [("COVID crash", "2020-02-20", "2020-04-30"),
          ("Ukraine war &<br>rate hikes", "2022-02-24", "2022-10-31")]
SILVER_PEAK = pd.Timestamp("2026-01-26")


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










def make_heatmap(r_table, p_table, cutoff, title, pair=None, heading="Which Assets Tend to Move Together?"):
    names = list(r_table.index)
    upper = np.triu(np.ones(r_table.shape, dtype=bool), k=1)
    sig = r_table.where(upper & (p_table.values < cutoff))
    ns = r_table.where(upper & (p_table.values >= cutoff)) * 0
    labels = sig.map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")
    hover = [[f"<b>{a} x {b}</b><br>r = {r_table.loc[a, b]:+.2f} <br>p = {p_table.loc[a, b]:.3f}"
              for b in names] for a in names]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=ns.values, x=names, y=names,
                              colorscale=[[0, NS_GREY], [1, NS_GREY]],
                              showscale=False,
                              xgap=3, ygap=3, hoverinfo="skip",
                              text=ns.map(lambda v: "" if pd.isna(v) else "n.s.").values, 
                              texttemplate="%{text}", textfont=dict(color="#4A5368", size=11)))


    fig.add_trace(go.Heatmap(z=sig.values, x=names, y=names, zmin=-1, zmax=1,
                             colorscale=GOLD_BLUE, xgap=3, ygap=3, 
                             text=labels.values, texttemplate="%{text}",
                             textfont=dict(size=14, color="white"),
                             customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                             colorbar=dict(thickness=10, len=0.85, outlinewidth=0, tickvals=[-1, 0, 1],
                                           ticktext=["-1 Opposite", "0 Unrelated", "+1 Together"],
                                           tickfont=dict(color=SOFT_TEXT, size=11))))

    for name in names:
        fig.add_annotation(x=name, y=name, text=f"<b>{NAMELAYOUT.get(name, name)}</b>",
                           showarrow=False, font=dict(size=13, color=LIGHT_TEXT))
    if pair and pair[0] != pair[1] and pair[0] in names and pair[1] in names:
        i, j = sorted([names.index(pair[0]), names.index(pair[1])])
        fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                        line=dict(color="white", width=2))
    subtitle = (f"<span style='font-size:13px;color:{SOFT_TEXT}'>{title}  |  "
                f"n.s. = Not Significant (p >= {cutoff:.2f})  |  Outlined = Pair Under Test</span>")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=80, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        title=dict(text=f"<b>{heading}</b><br>{subtitle}",
                                    x=0.01, font=dict(color=LIGHT_TEXT, size=20)))
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

def change_table(before, after):
    """How much each pair's correlation changed, and whether that change is significant."""
    r1 = before.corr().clip(-0.999, 0.999)
    r2 = after.corr().clip(-0.999, 0.999)
    z, p = fisher_z_test(r1, len(before), r2, len(after))
    p_table = pd.DataFrame(p, index=r1.index, columns=r1.columns)
    return r2 - r1, p_table, r1, r2

def make_change_heatmap(before, after, cutoff, title, pair=None, heading="Which Relationships Changed?"):
    diff, p_table, r1, r2 = change_table(before, after)
    names = list(diff.index)
    hover = [[f"<b>{a} × {b}</b><br>before: {r1.loc[a, b]:+.2f}<br>after: {r2.loc[a, b]:+.2f}"
              f"<br>change: {diff.loc[a, b]:+.2f} (p = {p_table.loc[a, b]:.3f})"
              for b in names] for a in names]
    fig = make_heatmap(diff, p_table, cutoff, title, pair, heading=heading)
    fig.data[1].update(zmin=-0.5, zmax=0.5, customdata=hover)
    fig.data[1].colorbar.update(tickvals=[-0.5, 0, 0.5],
                                ticktext=["Correlation Fell", "No Change", "Correlation Rose"])
    return fig

def ranking_table(before, after, cutoff):
    """One row per pair, the biggest significant changes first"""
    diff, p_table, r1, r2 = change_table(before, after)
    names = list(diff.index)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({"Asset 1": a, "Asset 2": b, "Before": r1.loc[a, b],
                         "After": r2.loc[a, b], "Change": diff.loc[a, b],
                         "p-value": p_table.loc[a, b]})
    table = pd.DataFrame(rows)
    table["Significant"] = table["p-value"] < cutoff
    return table.sort_values(["Significant", "Change"], ascending=False, 
                             key=lambda col: col.abs() if col.name == "Change" else col)
def colour_change(value):
    """Gold if the pair moved closer, blue if it moved apart; stronger = bigger change."""
    strength = min(abs(value) / 0.3, 1)
    rgb = "224,168,46" if value > 0 else "46,111,216"
    return f"background-color: rgba({rgb},{0.2 + 0.6 * strength:.2f}); color: white"

def fade_ns(row):
    """Grey out every cell in rows that aren't significant."""
    return ["color: #6B7280" if not row["Significant"] else ""] * len(row)

def rolling_corr(returns, a, b, window):
    """Correlation of a and b over the previous `window` trading days, day by day."""
    return returns[a].rolling(window).corr(returns[b]).dropna()

def confidence_band(rolling, window):
    """95% range for each rolling r, using the Fisher z trick."""
    z = np.arctanh(rolling.clip(-0.999, 0.999))
    se = 1 / np.sqrt(window - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

def show_rolling_stats(returns, a, b, window, start, end):
    """Five little stats to help simplify graph analysis. This time for the additional assets comparison."""
    rolling = rolling_corr(returns, a, b, window)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    before_r = rolling[rolling.index < start].mean()
    during_r = rolling[(rolling.index >= start) & (rolling.index <= end)].mean()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Latest", f"{rolling.iloc[-1]:+.2f}")
    m2.metric("Average Before Selected Period", f"{before_r:+.2f}" if pd.notna(before_r) else "-")
    m3.metric("Average in Selected Period", f"{during_r:+.2f}", f"{during_r - before_r:+.2f}", delta_color="off")
    m4.metric(f"Lowest ({rolling.idxmin():%b %Y})", f"{rolling.min():+.2f}")
    m5.metric(f"Highest ({rolling.idxmax():%b %Y})", f"{rolling.max():+.2f}")

def make_rolling_chart(returns, a, b, window, start, end, pair2=None):
    # (a) the band and the line(s)
    rolling = rolling_corr(returns, a, b, window)
    low, high = confidence_band(rolling, window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling.index, y=high, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rolling.index, y=low, line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(224,168,46,0.18)", name="95% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling, name=f"{a} x {b}",
                             line=dict(color=GOLD, width=2), hovertemplate="%{y:+.2f}"))
    if pair2:
        other = rolling_corr(returns, pair2[0], pair2[1], window)
        fig.add_trace(go.Scatter(x=other.index, y=other, name=f"{pair2[0]} x {pair2[1]}",
                                 line=dict(color=BLUE, width=2), hovertemplate="%{y:+.2f}"))

    # (b) shaded events and your period
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for label, s, e in EVENTS:
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(255,255,255,0.06)", line_width=0,
                      annotation_text=label, annotation_position="bottom left",
                      annotation_font=dict(color=SOFT_TEXT, size=10))
    fig.add_vrect(x0=start, x1=end, fillcolor="rgba(224,168,46,0.07)", line_width=0,
                  annotation_text="Your period", annotation_position="top right",
                  annotation_font=dict(color=GOLD, size=11))

    fig.add_vline(x=SILVER_PEAK, line_dash="dot", line_color="#C0C6D0", line_width=1.5)
    fig.add_annotation(x=SILVER_PEAK, y=-0.95, text="Silver peak", showarrow=False,
                       xanchor="right", font=dict(color=SOFT_TEXT, size=10))

    # (c) average before vs during, plus the zero line
    before = rolling[rolling.index < start]
    during = rolling[(rolling.index >= start) & (rolling.index <= end)]
    for part, dash in [(before, "dot"), (during, "dash")]:
        if len(part):
            fig.add_shape(type="line", x0=part.index[0], x1=part.index[-1], y0=part.mean(),
                          y1=part.mean(), line=dict(color=LIGHT_TEXT, width=1.5, dash=dash))
    fig.add_hline(y=0, line_color="#4A5368", line_width=1)

    # (d) styling, zoom buttons and range slider
    fig.update_layout(height=480, hovermode="x unified", margin=dict(l=10, r=10, t=120, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=LIGHT_TEXT),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        title=dict(x=0.01, text=f"<b>{a} x {b}: {window}-day Rolling Correlation</b><br>"
                   f"<span style='font-size:13px;color:{SOFT_TEXT}'>Shaded Band = 95% Confidence Interval   |   "
                   f"Dotted Lines = Average Before/During the Selected Period</span>"))
    fig.update_yaxes(range=[-1, 1], title="Correlation Coefficient (r)", gridcolor="#1C2230",
                     zeroline=False, automargin=True)
    fig.update_xaxes(gridcolor="#1C2230", rangeslider=dict(visible=True, thickness=0.06),
        rangeselector=dict(x=0, y=1.02, yanchor="bottom", bgcolor="#1C2230",
            activecolor="#2B3140", font=dict(color=LIGHT_TEXT),
            buttons=[dict(count=1, label="1Y", step="year", stepmode="backward"),
                     dict(count=3, label="3Y", step="year", stepmode="backward"),
                     dict(count=5, label="5Y", step="year", stepmode="backward"),
                     dict(step="all", label="All")]))
    return fig

def make_animated_heatmap(returns, assets, cutoff):
    years = sorted(returns.index.year.unique())
    figs = {}
    for year in years:
        r_table, p_table = correlation_table(returns.loc[str(year), assets])
        figs[year] = make_heatmap(r_table, p_table, cutoff, f"Year: {year}", heading="Year by Year: How the Links Shift")
    fig = go.Figure(figs[years[0]])
    fig.frames = [go.Frame(data=figs[y].data, name=str(y),
                           layout=dict(title=figs[y].layout.title)) for y in years]

    play = dict(frame=dict(duration=1500, redraw=True), transition=dict(duration=500), fromcurrent=True)
    stop = dict(frame=dict(duration=0, redraw=False), mode="immediate")
    fig.update_layout(height=550, margin=dict(l=10, r=10, t=80, b=10),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.02, xanchor="left", yanchor="top",
            bgcolor="#1C2230", font=dict(color=LIGHT_TEXT),
            buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                     dict(label="■ Pause", method="animate", args=[[None], stop])])],
        sliders=[dict(x=0.2, y=-0.02, len=0.8, currentvalue=dict(visible=False),
            font=dict(color=SOFT_TEXT),
            steps=[dict(label=str(y), method="animate", args=[[str(y)], stop]) for y in years])])
    return fig


def result_sentence(a, b, r1, r2, p, cutoff, start) -> str:
    """Summary of the Fisher Test result."""
    direction = "flipped" if np.sign(r1) != np.sign(r2) else "moved"
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    if p < cutoff:
        verdict = f"This change is statistically significant ({p_text})."
    else:
        verdict = f"This change could be chance ({p_text}), so there is no clear sign they decoupled."
    return (f"The {a} x {b} correlation coefficient {direction} from {r1:+.2f} before "
        f"{start:%b %Y} to {r2:+.2f} in this period. {verdict}")

def set_period(name, first, last):
    """Move the time slider to one of the group's shared periods."""
    split, peak = pd.Timestamp(SPLIT).date(), SILVER_PEAK.date()
    one_day = pd.Timedelta(days=1)
    periods = {"Before": (first, split - one_day), "Boom": (split, peak),
               "Crash": (peak + one_day, last), "2025 Onwards": (split, last)}
    st.session_state.p3_dates = periods[name]

def render():
    """Draw Panel 3 on the dashboard"""
    st.subheader("Correlation Heatmap")
    returns = get_returns().join(load_macro(), how="inner").dropna()

    # 1. Controls
    view = st.radio("View", ["Correlation Heatmap", "Rolling Graph", "Animated Timeline", "Decoupling Analysis"], horizontal=True, key="p3_view")
    if view == "Rolling Graph":
        assets = list(returns.columns)
        if "p3_assets" in st.session_state:
            st.session_state.p3_assets = st.session_state.p3_assets
    else:
        assets = st.multiselect("Assets", list(returns.columns),
                                default=["Gold", "Silver", "Oil", "S&P 500", "Real yield"],
                                key="p3_assets")
    first, last = returns.index.min().date(), returns.index.max().date()
    if view == "Animated Timeline":
        start, end = first, last
        if "p3_dates" in st.session_state:
            st.session_state.p3_dates = st.session_state.p3_dates
    else:
        if "p3_dates" not in st.session_state:
            st.session_state.p3_dates = (pd.Timestamp(SPLIT).date(), last)
        st.caption("Quick Period Select")
        for col, name in zip(st.columns(4), ["Before", "Boom", "Crash", "2025 Onwards"]):
                   col.button(name, key=f"p3_btn_{name}", on_click=set_period,
                   args=(name, first, last), width="stretch")
        start, end = st.slider("Time Period", min_value=first, max_value=last,
                           format="MMM YYYY", key="p3_dates")
    for key in ("p3_pair_a", "p3_pair_b"):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    if view == "Rolling Graph":
        cutoff = 0.05
        if "p3_cutoff" in st.session_state:
            st.session_state.p3_cutoff = st.session_state.p3_cutoff
    else:
        cutoff = st.slider("Significance Level", min_value=0.01, max_value=0.20,
                        value=0.05, step=0.01, format="%.2f", key="p3_cutoff")
    a = st.session_state.get("p3_pair_a", assets[0])
    b = st.session_state.get("p3_pair_b", assets[min(2, len(assets) - 1)])

                        

    # 2. Checking the choices make sense.
    period = returns.loc[str(start):str(end)]
    if len(assets) < 2 or len(period) < 30:
        st.warning("Please pick at least two assets and a period of at least 30 trading days to compare.")
        return

    # 3. Heatmap for the selected period.
    if view == "Correlation Heatmap":
        r_table, p_table = correlation_table(period[assets])
        title = f"{start:%b %Y} to {end:%b %Y}"
        st.plotly_chart(make_heatmap(r_table, p_table, cutoff, title, pair=(a,b)), width="stretch")
        st.caption(f"{len(period)} trading days. n.s. = not significant (p >= {cutoff}).")
    elif view == "Animated Timeline":
        st.plotly_chart(make_animated_heatmap(returns, assets, cutoff), width="stretch")
        st.caption("Press play to see how the correlations change year by year. "
                   "2026 covers January to October only. ")
                    
    elif view == "Rolling Graph":
        r1, r2 = st.columns(2)
        ra = r1.selectbox("Comparing:", assets, index=0, key="p3_roll_a")
        rb = r2.selectbox("Against:", assets, index=min(1, len(assets) - 1), key="p3_roll_b")
        if ra == rb:
            st.info("Pick two different assets for the rolling chart.")
        c1, c2 = st.columns(2)
        window = c1.slider("Smoothing: days per calculation", 30, 250, 90, step=10, key="p3_window",
                           help="Each point is the correlation over the previous N trading days. A bigger N makes a smoother line.")
        pairs = [f"{x} vs {y}" for i, x in enumerate(assets) for y in assets[i + 1:]]
        extra = c2.selectbox("Additional Asset Comparison:", ["None"] + pairs, key="p3_pair2")
        pair2 = None if extra == "None" else extra.split(" vs ")
        st.plotly_chart(make_rolling_chart(returns, ra, rb, window, start, end, pair2), width="stretch")
        st.markdown("#### Correlation Coefficient (r) Statistics")
        show_rolling_stats(returns, ra, rb, window, start, end)
        if pair2:
            st.markdown(f"##### Additional Assets: {pair2[0]} x {pair2[1]}")
            show_rolling_stats(returns, pair2[0], pair2[1], window, start, end)
    if view != "Decoupling Analysis":
        return

    # 3.5 Ranking every pair by how much it changed, using the fisher test.
    before_all = returns[returns.index < pd.Timestamp(start)]
    if len(before_all) >= 30:
        show_as = st.radio("Show as:", ["Table", "Heatmap"], horizontal=True, key="p3_rank_view")
        if show_as == "Heatmap":
            title = f"Before {start:%b %Y} vs {start:%b %Y} to {end:%b %Y}"
            st.plotly_chart(make_change_heatmap(before_all[assets], period[assets], cutoff, title, 
                                                pair=(a, b)), width="stretch")
            st.caption("Blue = Correlation Fell  |  Gold = Correlation Rose  |  Hover for before/after values.")
        else:
            with st.expander("Asset Pairs Ranked by Change", expanded=True):
                table = ranking_table(before_all[assets], period[assets], cutoff)
                st.caption(f"{int(table['Significant'].sum())} of {len(table)} pairs changed "
                           f"significantly (p < {cutoff}).")
                styled = (table.style.apply(fade_ns, axis=1)
                            .map(colour_change, subset=["Change"])
                            .format({"Before": "{:+.2f}", "After": "{:+.2f}", "Change": "{:+.2f}",
                                     "p-value": lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"}))
                st.dataframe(styled, hide_index=True, width="stretch",
                            column_config={"Significant": st.column_config.CheckboxColumn("Significant?")})
                st.download_button("Download as CSV", table.to_csv(index=False),
                                    file_name="correlation_changes.csv", key="p3_download")

    # 4. Fisher Test: The chosen period vs everything that comes before it.
    st.markdown("#### Pair in Focus: Real Change or Random Noise?")
    pick1, pick2 = st.columns(2)
    a = pick1.selectbox("Compare", assets, index=0, key="p3_pair_a")
    b = pick2.selectbox("with", assets, index=min(2, len(assets) - 1), key="p3_pair_b")
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
    c1.metric(f"{a} x {b} before {start:%b %Y}", f"r = {r1:+.2f}")
    c2.metric(f"{a} x {b} in this period", f"r = {r2:+.2f}", f"{r2 - r1:+.2f}", delta_color="off")
    c3.metric("p-value", "< 0.001" if p < 0.001 else f"{p:.3f}")
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
    fig = make_heatmap(r_table, p_table, 0.05, "")
    fig.update_layout(title_text="")
    fig.update_layout(height=370, margin=dict(l=0, r=0, t=10, b=0))
    fig.update_traces(textfont_size=11)
    fig.update_annotations(font_size=10)
    st.plotly_chart(fig, width="stretch", key="p3_preview", config={"displayModeBar": False})