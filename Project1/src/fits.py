# ols_fit, ridge_fit, fit_ols_SVD, 
# fit_ridge, predict, standardize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys 
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0,str(project_root))
from function_setups import design_matrix, runge, generate_data

def ols_fit(X, y, rcond=1e-12):
    """
        OLS via SVD (numerically stable, no normal equations):

            X = U diag(s) V^T
            θ = V diag(1/s_i) U^T y      with 1/s_i -> 0 for s_i < rcond * s_max

        Thresholding tiny singular values gives the minimum-norm least-squares
        solution when X is rank-deficient (which it is for high-degree
        polynomial fits on small n).
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    return Vt.T @ (s_inv * (U.T @ y))

def fit_ols_SVD(X,y,rcond = 1e-15):
    """
    OLS via SVD: theta = V diag(1/s_i) U^T y.
    Singular values below rcond * s_max are zeroed.
    """

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    return Vt.T @ (s_inv * (U.T @ y))

def fit_ridge(X,y,lam):
    """
    Ridge via SVD: theta = V diag(s_i / (s_i^2 + lam)) U^T y.
    This is the shrinkage view.
    """

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    shrink = s / (s ** 2 + lam)
    return Vt.T @ (shrink * (U.T @ y))

def predict(X, theta):
    """
    Linear prediction: y_hat = X @ theta.
    """
    y_pred = X @ theta
    return y_pred


def standardize_polynomials(X_train, X_test, intercept=True):
    """
        Standardise polynomial features column-wise.

        The intercept column (all ones) is left alone — standardising a
        constant is undefined. The scaler is fit on X_train only; X_test
        is transformed with the same means and stds. This is the correct
        workflow: no information from the test set enters the scaling.

        Edge case: if degree == 0 with intercept=True, there are no
        non-intercept columns to scale. Return the arrays unchanged and
        a None scaler. Callers must handle the None scaler.
    """
    if intercept:
        intercept_train = X_train[:, :1]
        intercept_test = X_test[:, :1]
        body_train = X_train[:, 1:]
        body_test = X_test[:, 1:]
    else:
        body_train = X_train
        body_test = X_test

    # --- Edge case: nothing to scale (degree 0 with intercept) ---
    if body_train.shape[1] == 0:
        return X_train.copy(), X_test.copy(), None

    scaler = StandardScaler()
    body_train_scaled = scaler.fit_transform(body_train)
    body_test_scaled = scaler.transform(body_test)

    if intercept:
        return (
            np.hstack([intercept_train, body_train_scaled]),
            np.hstack([intercept_test, body_test_scaled]),
            scaler,
        )
    return body_train_scaled, body_test_scaled, scaler

