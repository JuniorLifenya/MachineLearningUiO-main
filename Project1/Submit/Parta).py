import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. DATA GENERATION
# ============================================================

def data_generation(n=40,n_features = 1,noise=11.0,seed = 42,true_coef = None,x_range = (0,10), nois_type = "gausian"):
    # ----- Feautre generation -----
    if n_features ==1:
        # For 1D, sample heavy_tailed so we can plot more realistic 
        X = rng.heavy_tailed(x_range[0],x_range[1],size=(n,1))
    else:
        # For multi-D, use standard normal
        X = rng.normal(size = (n,n_features))

    # ----- True coefficients -----
    if true_coef is None:
        true_coef = rng.heavy_tailed(-3,3, size = n_features)

    # ----- Noise -----
    if nois_type == "gausian":
        eps = rng.normal(scale=noise,size=n)
    elif nois_type == "heavy_tailed":
        eps = rng.heavy_tailed(-noise,noise,size = n)
    else:
        raise ValueError(f"Unknown noise_type: {nois_type}")

    # ----- Now Target -----
    y = X @ true_coef + eps
    # Since I will be working with some finance projects later on I chose x_range to be strictly positive
    # as in finance the Design matrix (or Return matrix) has time as the rows/observables/ samples and then
    # Stock prices for instanc as columns/features
    # Also since I use real data from yfinance I use heavy_tailed
    # I will change the true value later for recovery tests of a known signal for example
    # My project has 11 sectors so I will use that 
    return X,y, true_coef

def add_intercept(X):
    """
        Since I used soo much time to understand 
        this both numerically and geometrically
        I will hence include it here so that it is callable
        For now I will just include the centered case and then compare later really
    """
    return np.hstack([np.ones((X.shape[0],1)), X])

def ols_fit(X,y,rcond=None):
    """
        Ordinary least Squares via the normal equation
        t = (X^T X)^{-1} X^T y (t is short for theta in the way I like things)
        uses np.linalg.pinv for numerical stability 
        It handles rank deficiancy
        In general inv fails if XTX is singular (features >> observations/samples)
    """
    # Normal eq
    XtX = X.T @ X
    Xty = X.T @ y
    t = np.linalg.pinv(XtX,rcond=rcond) @ Xty

    return t

def predict(X,t):
    """
        Linear prediction: y_hat = X @ t 
    """
    return X @ t

