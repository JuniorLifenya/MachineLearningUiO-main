"""
Part a): Data Generation and OLS
================================

Week 36 ML stack — Stage 1.

This module provides:
  - Synthetic data generation with configurable noise (Gaussian or Student-t).
  - OLS fitting via the normal equation with pseudoinverse for stability.
  - A minimal LinearModel interface that mirrors the sklearn API.
  - Evaluation metrics: MSE, RMSE, R².

Design notes
------------
- Heavy-tailed noise uses Student-t, the standard model for financial
  returns where extremes occur far more often than Gaussian predicts.
  Empirically confirmed in the Market Regime Detection project: every
  sector ETF shows excess kurtosis >> 0.
- `x_range` defaults to (0, 10) because financial features (prices,
  volumes, volatilities) are strictly positive. This is a modelling
  choice: finance data lives in the positive orthant, not on R^n.
- `LinearModel` mirrors sklearn so any fit function (OLS, Ridge, Lasso,
  custom) plugs into the same interface. This is the seam that lets
  Week 36 and the SVD regime project share downstream code.
- Centering: `add_intercept` is provided explicitly because in Week 36
  we proved that a centered X with uncentered y and no intercept cannot
  fit the mean of y. Making the choice explicit prevents silent bugs.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. DATA GENERATION
# ============================================================

def heavy_tailed_noise(rng, df, scale, size):
    """
    Student-t noise with given degrees of freedom, scaled.

    Parameters
    ----------
    rng : np.random.Generator
    df : float
        Degrees of freedom. df=3 -> very heavy tails; df=30 ~ Gaussian.
    scale : float
        Scale parameter (analogous to sigma for the Gaussian).
    size : int or tuple

    Returns
    -------
    ndarray
    """
    return rng.standard_t(df=df, size=size) * scale


def data_generation(
    n=40,
    n_features=1,
    noise=1.0,
    seed=42,
    true_coef=None,
    x_range=(0.0, 10.0),
    noise_type="gaussian",
    student_df=5.0,
):
    """
    Generate synthetic linear regression data.

    Model: y = X @ true_coef + eps

    Parameters
    ----------
    n : int
        Number of samples (observations / rows).
    n_features : int
        Number of features (columns).
    noise : float
        Noise magnitude. Interpreted as sigma for Gaussian, scale for
        Student-t.
    seed : int
        Random seed for reproducibility.
    true_coef : array-like or None
        True coefficients. If None, drawn uniformly from [-3, 3].
    x_range : tuple of float
        Range for feature sampling when n_features == 1. Defaults to
        strictly positive to reflect financial features.
    noise_type : str
        'gaussian' or 'student_t'.
    student_df : float
        Degrees of freedom for Student-t. Ignored if noise_type='gaussian'.

    Returns
    -------
    X : ndarray of shape (n, n_features)
    y : ndarray of shape (n,)
    true_coef : ndarray of shape (n_features,)
    """
    rng = np.random.default_rng(seed)

    # --- Features ---
    if n_features == 1:
        # Uniform over x_range for 1-D case, so plots are readable and
        # the fitted line has meaningful domain.
        X = rng.uniform(x_range[0], x_range[1], size=(n, 1))
    else:
        # Standard normal for multi-D. Design choice: our real features
        # will be standardized before entering the ML pipeline anyway.
        X = rng.normal(size=(n, n_features))

    # --- True coefficients ---
    if true_coef is None:
        true_coef = rng.uniform(-3.0, 3.0, size=n_features)
    else:
        true_coef = np.asarray(true_coef, dtype=float).ravel()
        if true_coef.shape[0] != n_features:
            raise ValueError(
                f"true_coef has length {true_coef.shape[0]}, "
                f"expected {n_features}"
            )

    # --- Noise ---
    if noise_type == "gaussian":
        eps = rng.normal(scale=noise, size=n)
    elif noise_type == "student_t":
        eps = heavy_tailed_noise(rng, df=student_df, scale=noise, size=n)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}. "
                         f"Use 'gaussian' or 'student_t'.")

    # --- Target ---
    y = X @ true_coef + eps

    return X, y, true_coef


def add_intercept(X):
    """
    Prepend a column of ones to X.

    Week 36 result: a centered X with no intercept cannot fit the mean
    of y. Including an explicit intercept column restores that capacity.
    """
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ============================================================
# 2. MODEL FITTING
# ============================================================

def ols_fit(X, y, rcond=None):
    """
    Ordinary Least Squares via the normal equation.

        theta = (X^T X)^{-1} X^T y

    Uses `np.linalg.pinv` for numerical stability. Unlike `inv`, pinv
    handles rank-deficient X^T X (e.g., more features than samples, or
    duplicated columns) by returning the minimum-norm solution.

    Notes
    -----
    - If X^T X is known invertible, `np.linalg.solve(XtX, Xty)` is faster.
    - If X is ill-conditioned, `np.linalg.lstsq(X, y, rcond=None)` is
      the standard numerical choice.
    """
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.pinv(XtX, rcond=rcond) @ Xty


def predict(X, theta):
    """Linear prediction: y_hat = X @ theta."""
    return X @ theta


class LinearModel:
    """
    Minimal linear model with an sklearn-style API.

    Fit functions plug in via the constructor, so the same interface
    serves OLS, Ridge, Lasso, or any custom estimator that returns a
    coefficient vector.
    """

    def __init__(self, fit_fn):
        self.fit_fn = fit_fn
        self.theta = None

    def fit(self, X, y):
        self.theta = self.fit_fn(X, y)
        return self

    def predict(self, X):
        if self.theta is None:
            raise RuntimeError("Model has not been fit yet. Call .fit(X, y).")
        return X @ self.theta


# ============================================================
# 3. EVALUATION METRICS
# ============================================================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def evaluate(y_true, y_pred, name="Model"):
    metrics = {
        "MSE":  mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2":   r2(y_true, y_pred),
    }
    print(f"{name}: MSE={metrics['MSE']:.4f}, "
          f"RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    return metrics


# ============================================================
# 4. DRIVER
# ============================================================

def _demo():
    """
    One figure, two panels: OLS without intercept vs with intercept.
    Reads the data, fits both, evaluates, plots.
    """
    # ---- 1. Generate data ----
    X, y, true_coef = data_generation(
        n=40,
        n_features=1,
        noise=1.0,
        seed=42,
        true_coef=np.array([2.5]),
        x_range=(0.0, 10.0),
        noise_type="gaussian",
    )
    print(f"True coefficient: {true_coef}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # ---- 2. Fit without intercept ----
    theta_no_int = ols_fit(X, y)
    y_pred_no_int = predict(X, theta_no_int)

    # ---- 3. Fit with intercept ----
    X_with_int = add_intercept(X)
    theta_with_int = ols_fit(X_with_int, y)
    y_pred_with_int = predict(X_with_int, theta_with_int)

    # ---- 4. Evaluate ----
    print("\n--- Without intercept ---")
    evaluate(y, y_pred_no_int, "OLS (no intercept)")

    print("\n--- With intercept ---")
    evaluate(y, y_pred_with_int, "OLS (with intercept)")

    # ---- 5. Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_sorted = np.sort(X, axis=0)

    axes[0].scatter(X, y, alpha=0.7, label="Data")
    axes[0].plot(
        x_sorted, predict(x_sorted, theta_no_int),
        color="red",
        label=f"OLS: y = {theta_no_int[0]:.2f} x",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("OLS without intercept")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(X, y, alpha=0.7, label="Data")
    axes[1].plot(
        x_sorted, predict(add_intercept(x_sorted), theta_with_int),
        color="green",
        label=(f"OLS: y = {theta_with_int[0]:.2f} "
               f"+ {theta_with_int[1]:.2f} x"),
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("OLS with intercept")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Part a) — Data Generation & OLS   |   "
        f"true slope = {true_coef[0]:.2f}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _demo()