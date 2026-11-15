import steamlit as st
impoprt numpy as np
import plotly.graph_objects as go
from scipy import stats

from data_loader import get_returns, SPLIT

def render():
    st.subheader("Distribution & Risk")

    returns = get_returns()

# MAKE 4 COLUNMS FOR PANEL CONTROLS 
col1, col2, col3, col4 = st.columns(4)
    asset = st.selectbox(
        "Asset",
        options=returns.columns,
        key="risk_asset"
    )

with col2:
     period = st.selectbox(
         "Period",
         options=["All data", "Since 2025", "Before 2025"],
         key="risk_period"
  )

with col3:
    frequency = st.radio(
        "Frequency",
        options=["Daily", "Weekly"],
        horizontal=True,
        key="risk_frequency"



with col4:
    confidence = st.selectbox(
        "VaR Confidence",
        options=[0.95, 0.99],
        format_func=lambda x: f"{int(x*100)}%",
        key="risk_confidence"
