import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from Part_ab import generate_data, runge, design_matrix


# ------------------------------------------------------------
# SVD fitters
# ------------------------------------------------------------

def fit_ols_SVD(X_train,y_train,intercept=False):
    """
        Perform the OLS fitting but with SVD for transparency and 
        Intuition/visualization. 
        t = V diag (1/s) U^T y 
    """
    U,s,Vt = np.linalg.svd(X, full_matrices=False)
    s_inv = np.array([1.0 / si if si > rcond * s[0] else 0.0 for si in si in s])

    return Vt.T @ (s_inv * (U.T @ y))

def fit_ridge_svd(X, y, lam):
    """
        Ridge via SVD:
            theta = V diag( s / (s^2 + lam)) U^T y 
    This is the shrinkage view from the lecture notes.
    """

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    shrink = s/ (s**2 + lam)

    return Vt.T @ (shrink * (U.T @ y ))

# ------------------------------------------------------------
# Fit/predict one polynomial degree
# ------------------------------------------------------------

def predict_deg_SVD(x_train, y_train, x_test, degree, lam = 0.0):
    """
    Fit polynomial degree d with OLS (lam=0) or Ridge (lam>0).

    Center y and stand polynomial features.
    Fit without explicit intercept column.
    This avoids penalising the intercept
    """
    if degree == 0:
        y_mean = y_train.mean()

        return np.full(x_train.shape, y_mean,) , np.full(x_test.shape,y_mean)

    X_train = design_matrix(x_train,degree,intercept=False)
    X_test = design_matrix(x_test,degree, intercept=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_mean = y_train.mean()
    y_train_c = y_train - y_mean

    if lam == 0.0:
        theta = fit_ols_SVD(X_train,y_train_c)
    else:
        theta = fit_ridge_svd(X_train,y_train_c,lam)

    yhat_train = X_train @ theta + y_mean
    yhat_test = X_test @ theta + y_mean

    return yhat_train, yhat_test

# ------------------------------------------------------------
# Sweep over degrees, repeated random splits
# ------------------------------------------------------------
