
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats

from data_loader import get_returns, SPLIT


def render():
    st.subheader("Distribution & Risk")

    returns = get_returns()

    # Dashboard controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        asset = st.selectbox(
            "Asset",
            options=returns.columns,
            key="risk_asset",
        )

    with col2:
        period = st.selectbox(
            "Period",
            options=["All data", "Since 2025", "Before 2025"],
            key="risk_period",
        )

    with col3:
        frequency = st.radio(
            "Frequency",
            options=["Daily", "Weekly"],
            horizontal=True,
            key="risk_frequency",
        )

    with col4:
        confidence = st.selectbox(
            "VaR Confidence",
            options=[0.95, 0.99],
            format_func=lambda x: f"{int(x * 100)}%",
            key="risk_confidence",
        )

    # Filter returns by period
    if period == "Since 2025":
        filtered_returns = returns[returns.index >= SPLIT]

    elif period == "Before 2025":
        filtered_returns = returns[returns.index < SPLIT]

    else:
        filtered_returns = returns.copy()

    # Select the chosen asset
    selected_returns = filtered_returns[asset]

    # Convert daily log returns into weekly returns
    if frequency == "Weekly":
        selected_returns = selected_returns.resample("W").sum(min_count=1)

    # Remove missing observations
    selected_returns = selected_returns.dropna()

    # Check that there are enough observations
    if len(selected_returns) < 30:
        st.warning("Not enough observations to calculate reliable statistics.")
        return

    # Calculate distribution statistics
    mean_return = selected_returns.mean()
    volatility = selected_returns.std()
    skewness = stats.skew(selected_returns)
    kurtosis = stats.kurtosis(selected_returns)

    # Calculate historical Value at Risk
    var_threshold = np.percentile(
        selected_returns,
        (1 - confidence) * 100
    )

    # Display VaR as a positive loss
    var = abs(var_threshold)

    # Display key statistics
    stat1, stat2, stat3, stat4, stat5 = st.columns(5)

    stat1.metric(
        "Mean Return",
        f"{mean_return:.2%}"
    )

    stat2.metric(
        "Volatility",
        f"{volatility:.2%}"
    )

    stat3.metric(
        "Skewness",
        f"{skewness:.2f}"
    )

    stat4.metric(
        "Excess Kurtosis",
        f"{kurtosis:.2f}",
        help="Measures how heavy the tails are compared with a normal distribution. Zero represents a normal distribution."
    )

    stat5.metric(
        f"{int(confidence * 100)}% VaR",
        f"{var:.2%}",
        help="Historical estimate of the loss threshold at the selected confidence level."
    )

    # Set up points for the normal distribution curve
    x_values = np.linspace(
        selected_returns.min(),
        selected_returns.max(),
        200
    )

    normal_curve = stats.norm.pdf(
        x_values,
        loc=mean_return,
        scale=volatility
    )

    # Create the chart
    fig = go.Figure()

    # Histogram of actual returns
    fig.add_trace(
        go.Histogram(
            x=selected_returns,
            histnorm="probability density",
            name=f"{asset} returns",
            opacity=0.7,
            nbinsx=40
        )
    )

    # Normal distribution for comparison
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=normal_curve,
            mode="lines",
            name="Normal distribution",
            line=dict(width=2)
        )
    )

    # Mark the Value at Risk threshold
    fig.add_vline(
        x=var_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"{int(confidence * 100)}% VaR",
        annotation_position="top"
    )

    # Chart layout
    fig.update_layout(
        title=f"{asset}: Return Distribution and Risk",
        xaxis_title="Log Return",
        yaxis_title="Density",
        barmode="overlay",
        hovermode="x unified",
        legend_title="Series",
        height=450
    )

    # Display the interactive chart
    st.plotly_chart(fig, use_container_width=True)

    # Summary below the chart
    st.caption(
        f"Based on {len(selected_returns)} {frequency.lower()} observations. "
        f"The {int(confidence * 100)}% historical VaR is {var:.2%}."
    )
