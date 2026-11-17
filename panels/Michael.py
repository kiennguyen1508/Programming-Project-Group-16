
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from datetime import datetime, timedelta
from pathlib import Path
import requests
import io


# settings
TRADING_DAYS = 252
START_DATE = "2015-01-01"
SPLIT_DATE = "2025-01-01"

GOLD_COLOUR = "#D4A017"
SP500_COLOUR = "#2563EB"

DATA_FOLDER = Path("data")
DATA_FOLDER.mkdir(exist_ok=True)

# extra data sources
FRED_DFII10 = "DFII10"
FRED_GVZ = "GVZCLS"

SKEW_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv"
)
