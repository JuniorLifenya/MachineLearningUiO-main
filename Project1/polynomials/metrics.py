import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys 
from pathlib import Path



project_root = Path(__file__).parent.parent.parent
sys.path.insert(0,str(project_root))
from src.function_setups import design_matrix, runge, generate_data
from src.fits import fit_ols_SVD, fit_ridge, predict, standardize_polynomials


y_pred = predict(X_test, theta)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def experiment_ols_degree(x, y, degrees, test_size=0.3, seed=42,
                          scale=True, intercept=True):
    """
      Sweep polynomial degree for OLS.
      For each degree:
      1. Build the polynomial design matrix.
      2. Train/test split.
      3. Optionally standardise features (fit on train only).
      4. Fit OLS, predict on train and test.
      5. Record MSE and R^2 for both sets, plus the coefficient vector.
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

        theta = fit_ols_SVD(X_train, y_train)
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


def experiment_ridge(x, y, degrees, lambdas, test_size=0.3, seed=42,
                     scale=True, intercept=True):
    """
        Sweep polynomial degree AND λ for Ridge.
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
            theta = fit_ridge(X_train, y_train, lam=lam)
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

        theta = fit_ols_SVD(X_train_s, y_train)
        ax.plot(xx, X_xx_s @ theta, color=color, linewidth=1.5,
                label=f"degree {d}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("OLS fits at three degrees (scaled)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Coefficient magnitudes vs degree ---
    ax = axes[1, 1]
    thetas = np.array([np.pad(t, (0, max_deg + 1 - len(t)), constant_values=np.nan)
                    for t in res_scaled["theta_train"]])
    for j in range(thetas.shape[1]):
        ax.plot(res_scaled["degrees"], thetas[:, j], marker="o", markersize=3,
                linewidth=1, alpha=0.7, label=f"$\\theta_{{{j}}}$")
    ax.set_yscale("symlog", linthresh=1e-2)   # signed, handles zeros
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"$\theta_j$")
    ax.set_title("Coefficients vs degree")
    ax.legend(fontsize=7, ncol=2, frameon=False)
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

    theta_ols = fit_ols_SVD(X_train_s, y_train)
    ax.plot(xx, X_xx_s @ theta_ols, color="#e74c3c", linewidth=1.4,
            label="OLS")

    for lam, color in zip([1e-4, 1e-2, 1.0], ["#2ecc71", "#f39c12", "#9b59b6"]):
        theta_r = fit_ridge(X_train_s, y_train, lam=lam)
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

    