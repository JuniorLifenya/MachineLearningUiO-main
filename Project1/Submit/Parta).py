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

    # Instead np.lingalg.lstsq(X,y) is also more numerically stable and standard

    # If XTX is invertible then np.linalg.solve(XtX,Xty)
    # Faster...will check for my project really

    return t

def predict(X,t):
    """
        Linear prediction: y_hat = X @ t 
    """
    return X @ t

# ----- Small interface so that I can change models more easily -----
class LinearModel:
    """
        Base linear model: y = X @ theta.
    """
    def __init__(self,fit_fn):
        self.fit_fn = fit_fn
        self.theta = None
    def fit(self,X,y):
        self.theta = self.fit_fn(X,y)
        return self 
    def prediction(self,X):
        return X @ self.theta

# Usage
ols = LinearModel(ols_fit).fit(X_train,y_train)
y_pred = ols.predicti(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def evaluate(y_true, y_pred, name="Model"):
    metrics = {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
    print(f"{name}: MSE={metrics['MSE']:.4f}, "
          f"RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    return metrics

 # --- 6. Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Without intercept
axes[0].scatter(X, y, alpha=0.7, label="Data")
x_sorted = np.sort(X, axis=0)
axes[0].plot(x_sorted, predict(x_sorted, theta_no_int),
                 color="red", label=f"OLS: y = {theta_no_int[0]:.2f} x")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("OLS without intercept")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# With intercept
axes[1].scatter(X, y, alpha=0.7, label="Data")
axes[1].plot(x_sorted, predict(add_intercept(x_sorted), theta_with_int),
                 color="green",
                 label=f"OLS: y = {theta_with_int[0]:.2f} + {theta_with_int[1]:.2f} x")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("OLS with intercept")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



