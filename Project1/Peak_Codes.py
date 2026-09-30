"""
Project 1 — reusable core for Parts a)-d).

Core workflow:
    data -> split -> represent -> scale -> fit -> predict -> evaluate
    bootstrap -> estimate prediction variability
    cross-validation -> select model/hyperparameters
"""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# ============================================================
# DATA
# ============================================================

def runge(x):
    """Runge function f(x) = 1/(1+25x^2)."""
    x = np.asarray(x)
    return 1.0 / (1.0 + 25.0 * x**2)


def generate_data(n=100, sigma=0.1, seed=2026):
    """Return x and y = Runge(x) + Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    y = runge(x) + rng.normal(0.0, sigma, n)
    return x, y


# ============================================================
# DESIGN MATRIX
# ============================================================

def design_matrix(x, degree, intercept=True):
    """
    Columns:
        [1, x, x^2, ..., x^degree]       if intercept=True
        [x, x^2, ..., x^degree]          if intercept=False
    """
    x = np.asarray(x).ravel()
    start = 0 if intercept else 1
    return np.column_stack(
        [x**p for p in range(start, degree + 1)]
    )


# ============================================================
# METRICS
# ============================================================

def mse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.mean((y_true - y_pred)**2)


def r2_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)

    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ============================================================
# DATA SPLITTING
# ============================================================

def split_data(x, y, test_size=0.2, seed=2026):
    """Split x and y together into train/test sets."""
    return train_test_split(
        x, y,
        test_size=test_size,
        random_state=seed
    )


# ============================================================
# SVD FITTERS
# ============================================================

def fit_ols_svd(X, y, rcond=None):
    """
    OLS using the SVD directly.
        X = U diag(s) V^T
        X^+ = V diag(1/s) U^T
        theta = X^+ y
    """