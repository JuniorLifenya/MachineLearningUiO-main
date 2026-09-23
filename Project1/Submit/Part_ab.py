"""
Project 1, Parts a) and b): OLS and Ridge on Runge's function
=============================================================

FYS-STK3155/FYS4155, Fall 2026.

This module covers:
  Part a) — OLS with polynomial features up to degree 15+, with
            scaling/centering and a train/test split. Reports MSE and R²
            as functions of polynomial degree, and plots the fitted
            coefficients θ against degree.
  Part b) — Ridge regression with the same pipeline, sweeping λ.
            Connects the results to the SVD shrinkage of singular-value
            modes discussed in Chapter 3 of the lecture notes.

Some notes
------------
- Runge's function f(x) = 1/(1 + 25x²) on [-1, 1] is the testbed.
  It is smooth but has strong curvature near |x| = 1; high-degree
  polynomials fitted on uniform points diverge wildly at the edges
  (Runge phenomenon). This is exactly what makes it a good case study
  for regularisation.
- Polynomial features [1, x, x², ..., x^d] have wildly different
  scales: x^15 ranges over [-1, 1] but is tiny in magnitude almost
  everywhere. Without standardisation, X^T X is ill-conditioned and
  the normal equation becomes unstable for d ≳ 8. Standardisation
  (centering,divide by std, per column, fitted on train only)
  fixes this.
- Scaling is done INSIDE the train/test split: the scaler is fit on
  the training set only, then applied to the test set. Fitting on all
  data leaks test information into training and biases the test error.
- Ridge closes the gap when OLS blows up: for large d, Ridge with
  the right λ recovers a stable solution while OLS returns a wild
  coefficient vector dominated by noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# 1. THE FUNCTION AND DESIGN MATRIX
# ============================================================

def runge(x):
    """Runge's function: f(x) = 1 / (1 + 25 x²)."""
    return 1.0 / (1.0 + 25.0 * x ** 2)


def design_matrix(x, degree, intercept=True):
    """
    Polynomial design matrix.

    Columns are [1, x, x², ..., x^degree] if intercept=True,
    or [x, x², ..., x^degree] if intercept=False.

    Parameters
    ----------
    x : ndarray of shape (n,)
    degree : int
    intercept : bool

    Returns
    -------
    X : ndarray of shape (n, degree + 1) or (n, degree)
    """
    x = np.asarray(x).ravel()
    start = 0 if intercept else 1
    return np.vstack([x ** p for p in range(start, degree + 1)]).T


def generate_data(n=100, sigma=0.1, seed=2026, x_min=-1.0, x_max=1.0):
    """
    Sample Runge's function with additive Gaussian noise.

    Parameters
    ----------
    n : int
        Number of samples.
    sigma : float
        Noise standard deviation.
    seed : int
    x_min, x_max : float
        Domain boundaries. Defaults to [-1, 1].

    Returns
    -------
    x : ndarray of shape (n,)
    y : ndarray of shape (n,)
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(x_min, x_max, n))
    y = runge(x) + rng.normal(0.0, sigma, n)
    return x, y


# ============================================================
# 2. MODEL FITTING
# ============================================================

def ols_fit(X, y, rcond=None):
    """
        OLS via the normal equation, solved with the pseudoinverse.
            θ = (X^T X)^{-1} X^T y
        `pinv` handles rank-deficient X^T X gracefully, which matters for
        high-degree polynomial fits on small samples.
        We try SVD for transparancy really
    """
    U,s,Vt = np.linalg.svd(X, full_matrices= False)
    
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.pinv(XtX, rcond=rcond) @ Xty


def ridge_fit(X, y, lam=1.0):
    """
    Ridge regression via the normal equation.

        θ = (X^T X + λ I)^{-1} X^T y

    Note: we do NOT penalise the intercept term when intercept=True,
    i.e. we do not add λ to the (0, 0) entry of the regularisation
    matrix. This is the standard convention (sklearn's
    Ridge(fit_intercept=True) does the same). If you include the
    intercept column and penalise it, the model can be forced to
    shrink its own mean, which is usually not what you want.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix, possibly with intercept column at index 0.
    y : ndarray of shape (n,)
    lam : float
        Regularisation strength.

    Returns
    -------
    theta : ndarray of shape (p,)
    """
    n_features = X.shape[1]
    I = np.eye(n_features)
    # Do not penalise the intercept (assumed at column 0).
    I[0, 0] = 0.0
    return np.linalg.pinv(X.T @ X + lam * I) @ X.T @ y


def predict(X, theta):
    """Linear prediction: y_hat = X @ theta."""
    return X @ theta


# ============================================================
# 3. SCALING (fit on train, apply to test)
# ============================================================

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

    Parameters
    ----------
    X_train, X_test : ndarray
    intercept : bool

    Returns
    -------
    X_train_scaled, X_test_scaled : ndarray
    scaler : StandardScaler or None
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
# ============================================================
# 4. METRICS
# ============================================================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ============================================================
# 5. EXPERIMENT: OLS OVER POLYNOMIAL DEGREE (Part a)
# ============================================================

def experiment_ols_degree(x, y, degrees, test_size=0.3, seed=42,
                          scale=True, intercept=True):
    """
    Sweep polynomial degree for OLS.

    For each degree:
      1. Build the polynomial design matrix.
      2. Train/test split.
      3. Optionally standardise features (fit on train only).
      4. Fit OLS, predict on train and test.
      5. Record MSE and R² for both sets, plus the coefficient vector.

    Returns
    -------
    dict with arrays: degrees, mse_train, mse_test, r2_train, r2_test,
    and a list of coefficient vectors indexed by degree.
    """
    results = {
        "degrees": [],
        "mse_train": [],
        "mse_test": [],
        "r2_train": [],
        "r2_test": [],
        "theta_train": [],
        "n_features": [],
    }

    for d in degrees:
        X = design_matrix(x, d, intercept=intercept)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed,
        )

        if scale:
            X_train, X_test, _ = standardize_polynomials(
                X_train, X_test, intercept=intercept,
            )

        theta = ols_fit(X_train, y_train)
        y_pred_train = predict(X_train, theta)
        y_pred_test = predict(X_test, theta)

        results["degrees"].append(d)
        results["mse_train"].append(mse(y_train, y_pred_train))
        results["mse_test"].append(mse(y_test, y_pred_test))
        results["r2_train"].append(r2(y_train, y_pred_train))
        results["r2_test"].append(r2(y_test, y_pred_test))
        results["theta_train"].append(theta)
        results["n_features"].append(X.shape[1])

    for key in ["degrees", "mse_train", "mse_test",
                "r2_train", "r2_test", "n_features"]:
        results[key] = np.array(results[key])

    return results


# ============================================================
# 6. EXPERIMENT: RIDGE OVER DEGREE AND LAMBDA (Part b)
# ============================================================

def experiment_ridge(x, y, degrees, lambdas, test_size=0.3, seed=42,
                     scale=True, intercept=True):
    """
    Sweep polynomial degree AND λ for Ridge.

    Returns
    -------
    dict with:
      degrees        : (D,)
      lambdas        : (L,)
      mse_test       : (D, L)
      r2_test        : (D, L)
      theta          : list of (D, L) arrays of coefficient vectors
    """
    D = len(degrees)
    L = len(lambdas)
    mse_test = np.zeros((D, L))
    r2_test = np.zeros((D, L))
    theta_all = np.empty((D, L), dtype=object)

    for i, d in enumerate(degrees):
        X = design_matrix(x, d, intercept=intercept)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed,
        )
        if scale:
            X_train, X_test, _ = standardize_polynomials(
                X_train, X_test, intercept=intercept,
            )

        for j, lam in enumerate(lambdas):
            theta = ridge_fit(X_train, y_train, lam=lam)
            y_pred_test = predict(X_test, theta)
            mse_test[i, j] = mse(y_test, y_pred_test)
            r2_test[i, j] = r2(y_test, y_pred_test)
            theta_all[i, j] = theta

    return {
        "degrees": np.array(degrees),
        "lambdas": np.array(lambdas),
        "mse_test": mse_test,
        "r2_test": r2_test,
        "theta": theta_all,
    }


# ============================================================
# 7. PLOTTING
# ============================================================

def plot_part_a(x, y, res_scaled, res_unscaled=None, savepath=None):
    """
    Four panels telling the Part a) story:
      [0] MSE vs degree, scaled features (train and test)
      [1] R² vs degree, scaled features (train and test)
      [2] Fitted curves at three representative degrees
      [3] Coefficient magnitudes vs degree (scaled features)
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- Panel 0: MSE vs degree ---
    ax = axes[0, 0]
    ax.semilogy(res_scaled["degrees"], res_scaled["mse_train"],
                "o-", label="Train", color="#3498db")
    ax.semilogy(res_scaled["degrees"], res_scaled["mse_test"],
                "s-", label="Test", color="#e74c3c")
    if res_unscaled is not None:
        ax.semilogy(res_unscaled["degrees"], res_unscaled["mse_test"],
                    "^--", label="Test (unscaled)", color="#95a5a6",
                    alpha=0.7)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("MSE vs degree — scaled features")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    # --- Panel 1: R² vs degree ---
    ax = axes[0, 1]
    ax.plot(res_scaled["degrees"], res_scaled["r2_train"],
            "o-", label="Train", color="#3498db")
    ax.plot(res_scaled["degrees"], res_scaled["r2_test"],
            "s-", label="Test", color="#e74c3c")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("R²")
    ax.set_title("R² vs degree — scaled features")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Fitted curves ---
    ax = axes[1, 0]
    xx = np.linspace(x.min(), x.max(), 400)
    ax.plot(xx, runge(xx), "k-", linewidth=2, label="Runge")
    ax.scatter(x, y, s=10, color="#BB5566", alpha=0.5, label="Data")
    for d, color in zip([3, 8, 15], ["#2ecc71", "#f39c12", "#9b59b6"]):
        X_all = design_matrix(x, d)
        X_xx = design_matrix(xx, d)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=0.3, random_state=42,
        )
        X_train_s, X_test_s, scaler = standardize_polynomials(X_train, X_test)
        # For prediction on xx, apply the same scaling
        body_xx = X_xx[:, 1:]
        body_xx_s = scaler.transform(body_xx)
        X_xx_s = np.hstack([X_xx[:, :1], body_xx_s])

        theta = ols_fit(X_train_s, y_train)
        ax.plot(xx, X_xx_s @ theta, color=color, linewidth=1.5,
                label=f"degree {d}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("OLS fits at three degrees (scaled)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Coefficient magnitudes vs degree ---
    ax = axes[1, 1]
    max_deg = res_scaled["degrees"].max()
    for i, d in enumerate(res_scaled["degrees"]):
        theta = res_scaled["theta_train"][i]
        ax.scatter([d] * len(theta), np.abs(theta),
                   s=8, color="#34495e", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("|θ_j| (log scale)")
    ax.set_title("Coefficient magnitudes vs degree")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Part a) — OLS on Runge's function",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140, bbox_inches="tight")
    plt.show()


def plot_part_b(x, y, ridge_res, ols_res, savepath=None):
    """
    Four panels for Part b):
      [0] Test MSE heat map over (degree, λ)
      [1] Test MSE vs λ at three degrees
      [2] Coefficient shrinkage vs λ at fixed degree
      [3] Ridge fit vs OLS at the degree where OLS diverges
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    degrees = ridge_res["degrees"]
    lambdas = ridge_res["lambdas"]

    # --- Panel 0: Heat map ---
    ax = axes[0, 0]
    im = ax.imshow(
        np.log10(ridge_res["mse_test"]),
        aspect="auto", origin="lower",
        extent=[np.log10(lambdas[0]), np.log10(lambdas[-1]),
                degrees[0], degrees[-1]],
        cmap="viridis",
    )
    ax.set_xlabel(r"$\log_{10}\lambda$")
    ax.set_ylabel("Polynomial degree")
    ax.set_title("Test MSE (log₁₀) over (degree, λ)")
    plt.colorbar(im, ax=ax, label=r"$\log_{10}$ MSE")

    # --- Panel 1: MSE vs λ at three degrees ---
    ax = axes[0, 1]
    for d, color in zip([3, 8, 15], ["#2ecc71", "#f39c12", "#9b59b6"]):
        i = np.where(degrees == d)[0][0]
        ax.loglog(lambdas, ridge_res["mse_test"][i, :],
                  "o-", color=color, label=f"degree {d}", markersize=4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Test MSE")
    ax.set_title("Test MSE vs λ at three degrees")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, which="both")

    # --- Panel 2: Coefficient shrinkage vs λ ---
    # Pick the highest degree and show each coefficient's magnitude
    ax = axes[1, 0]
    d = degrees[-1]
    i = np.where(degrees == d)[0][0]
    theta_grid = np.array([ridge_res["theta"][i, j]
                           for j in range(len(lambdas))])
    for k in range(theta_grid.shape[1]):
        ax.loglog(lambdas, np.abs(theta_grid[:, k]),
                  color="#34495e", alpha=0.5, linewidth=0.8)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$|\theta_j|$")
    ax.set_title(f"Shrinkage of coefficients vs λ (degree {d})")
    ax.grid(True, alpha=0.3, which="both")

    # --- Panel 3: Ridge vs OLS at the divergence degree ---
    ax = axes[1, 1]
    xx = np.linspace(x.min(), x.max(), 400)
    ax.plot(xx, runge(xx), "k-", linewidth=2, label="Runge")
    ax.scatter(x, y, s=10, color="#BB5566", alpha=0.5, label="Data")

    d = 15
    X = design_matrix(x, d)
    X_xx = design_matrix(xx, d)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    X_train_s, X_test_s, scaler = standardize_polynomials(X_train, X_test)
    body_xx = X_xx[:, 1:]
    X_xx_s = np.hstack([X_xx[:, :1], scaler.transform(body_xx)])

    theta_ols = ols_fit(X_train_s, y_train)
    ax.plot(xx, X_xx_s @ theta_ols, color="#e74c3c", linewidth=1.4,
            label="OLS")

    for lam, color in zip([1e-4, 1e-2, 1.0], ["#2ecc71", "#f39c12", "#9b59b6"]):
        theta_r = ridge_fit(X_train_s, y_train, lam=lam)
        ax.plot(xx, X_xx_s @ theta_r, color=color, linewidth=1.4,
                label=rf"Ridge $\lambda={lam}$")

    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"OLS vs Ridge at degree {d}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Part b) — Ridge on Runge's function",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=140, bbox_inches="tight")
    plt.show()


# ============================================================
# 8. DRIVERS
# ============================================================

def demo_part_a():
    print("=" * 65)
    print("PART A — OLS on Runge's function")
    print("=" * 65)

    # --- Data ---
    x, y = generate_data(n=100, sigma=0.1, seed=2026)
    print(f"n = {len(x)}, sigma = 0.1, domain = [{x.min():.2f}, {x.max():.2f}]")

    # --- Degree sweep, scaled vs unscaled ---
    degrees = list(range(0, 16))
    res_scaled = experiment_ols_degree(x, y, degrees, scale=True)
    res_unscaled = experiment_ols_degree(x, y, degrees, scale=False)

    print(f"\n{'d':>3} {'n_feat':>7} "
          f"{'MSE_train':>12} {'MSE_test':>12} {'R2_test':>10} "
          f"{'MSE_test_unscaled':>20}")
    for i, d in enumerate(degrees):
        print(f"{d:>3} {res_scaled['n_features'][i]:>7} "
              f"{res_scaled['mse_train'][i]:>12.4e} "
              f"{res_scaled['mse_test'][i]:>12.4e} "
              f"{res_scaled['r2_test'][i]:>10.4f} "
              f"{res_unscaled['mse_test'][i]:>20.4e}")

    print("\nInterpretation prompts for your report:")
    print("  - At which degree does MSE_test start rising?")
    print("  - Does R2_test go negative? At what degree?")
    print("  - Compare the scaled and unscaled test MSEs. Which is better?")
    print("  - Look at |theta_j| growth in the last panel: is it monotone?")

    plot_part_a(x, y, res_scaled, res_unscaled=res_unscaled)

    return res_scaled, res_unscaled


def demo_part_b():
    print("\n" + "=" * 65)
    print("PART B — Ridge on Runge's function")
    print("=" * 65)

    x, y = generate_data(n=100, sigma=0.1, seed=2026)

    degrees = list(range(0, 16))
    lambdas = np.logspace(-6, 2, 40)

    ridge_res = experiment_ridge(x, y, degrees, lambdas)

    # --- Best (degree, λ) ---
    i_best, j_best = np.unravel_index(
        np.argmin(ridge_res["mse_test"]), ridge_res["mse_test"].shape,
    )
    d_best = ridge_res["degrees"][i_best]
    lam_best = ridge_res["lambdas"][j_best]
    mse_best = ridge_res["mse_test"][i_best, j_best]
    print(f"\nBest Ridge: degree={d_best}, lambda={lam_best:.3e}, "
          f"test MSE={mse_best:.4e}")

    # --- Reference OLS at same degrees ---
    ols_res = experiment_ols_degree(x, y, degrees, scale=True)
    i_ols_best = np.argmin(ols_res["mse_test"])
    print(f"Best OLS:   degree={degrees[i_ols_best]}, "
          f"test MSE={ols_res['mse_test'][i_ols_best]:.4e}")

    # --- Tables ---
    print("\nTest MSE for selected (degree, lambda):")
    print(f"{'d':>3} " + " ".join(f"{lam:>10.1e}" for lam in
                                    [1e-4, 1e-2, 1e-1, 1.0]))
    for d in [3, 5, 8, 10, 12, 15]:
        i = np.where(ridge_res["degrees"] == d)[0][0]
        row = []
        for lam in [1e-4, 1e-2, 1e-1, 1.0]:
            j = np.argmin(np.abs(ridge_res["lambdas"] - lam))
            row.append(f"{ridge_res['mse_test'][i, j]:>10.3e}")
        print(f"{d:>3} " + " ".join(row))

    print("\nInterpretation prompts for your report:")
    print("  - For fixed degree, how does test MSE vary with lambda?")
    print("    Is there an optimal lambda per degree?")
    print("  - At high degree, how much does Ridge rescue OLS?")
    print("  - Coefficient panel: which coefficients shrink first as lambda grows?")
    print("  - SVD view: small singular values of X correspond to directions")
    print("    where Ridge shrinks most aggressively. Can you verify numerically?")

    plot_part_b(x, y, ridge_res, ols_res)

    return ridge_res, ols_res


def verify_svd_shrinkage():
    """
    Optional sanity check: confirm the SVD view of Ridge.

    Ridge solution in SVD coordinates:
        theta = V @ diag(sigma_i / (sigma_i^2 + lambda)) @ U^T @ y
    This should match the closed-form (X^T X + lambda I)^{-1} X^T y
    to machine precision.
    """
    x, y = generate_data(n=80, sigma=0.1, seed=1)
    X = design_matrix(x, 10)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=0,
    )
    X_train_s, _, _ = standardize_polynomials(X_train, X_train)
    lam = 0.1

    theta_closed = ridge_fit(X_train_s, y_train, lam=lam)

    U, S, Vt = np.linalg.svd(X_train_s, full_matrices=False)
    shrink = S / (S ** 2 + lam)
    theta_svd = Vt.T @ (shrink * (U.T @ y_train))

    err = np.max(np.abs(theta_closed - theta_svd))
    print(f"Ridge SVD check: max|theta_closed - theta_svd| = {err:.2e}")


if __name__ == "__main__":
    demo_part_a()
    demo_part_b()
    print("\n--- SVD shrinkage verification ---")
    verify_svd_shrinkage()