"""
Panel: Silver vs gold regression (Charlie).
"""
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

from data_loader import get_returns

PERIODS = {"2015-2024": ("2015-01-01", "2024-12-31"),
           "2025": ("2025-01-01", "2025-12-31"),
           "2026": ("2026-01-01", None)}

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

def render():
    """
    Draw my panel inside the dashboard.
    """
    st.subheader("Silver vs gold regression")
    st.write("Coming soon")
    