import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets

# ============================================================
# 1. THE FUNCTION AND DESIGN MATRIX
# ============================================================

def runge(x):
    """
    Runge's function: f(x) = 1 / (1 + 25 x²).
    """
    return 1.0 / (1.0 + 25.0 * x ** 2)

# ============================================================
# Activation functions using standard NumPy (1:1 drop-in for jnp)
# ============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def ReLU(z):
    return np.where(z > 0, z, 0.0)

def ReLU_der(z):
    return np.where(z > 0, 1.0, 0.0)

def leaky_ReLU(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_ReLU_der(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

# ============================================================
# Fitting Codes 
# ============================================================

def fit_ols_SVD(X, y, rcond=1e-15):
    """
    OLS via SVD: theta = V diag(1/s_i) U^T y.
    Singular values below rcond * s_max are zeroed.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    return Vt.T @ (s_inv * (U.T @ y))

def fit_ridge(X, y, lam):
    """
    Ridge via SVD: theta = V diag(s_i / (s_i^2 + lam)) U^T y.
    This is the shrinkage view.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    shrink = s / (s ** 2 + lam)
    return Vt.T @ (shrink * (U.T @ y))

# =========================================================================================

if __name__ == "__main__":
    
    # Ensure you have your design_matrix function imported or defined correctly
    try:
        from data_aquisition import design_matrix
    except ImportError:
        # Fallback polynomial design matrix if data_aquisition is missing locally
        def design_matrix(x, degree=18, intercept=True):
            X = np.vander(x, degree + 1, increasing=True)
            if not intercept:
                X = X[:, 1:]
            return X

    rng = np.random.default_rng(2026)
    n = 100
    sigma = 0.1
    x = np.sort(rng.uniform(-1, 1, n))
    y = runge(x) + rng.normal(0, sigma, n)

    xx = np.linspace(-1, 1, 400)
    plt.figure(figsize=(8, 5))
    plt.plot(xx, runge(xx), color="green", label="Runge function fitting test")
    plt.scatter(x, y, s=12, color="orange", label=f"data, $\sigma = {sigma}$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(frameon=True)
    plt.show()

    X = design_matrix(x, degree=18, intercept=True)
    cond_xtx = np.linalg.cond(X.T @ X)
    exponent = np.log10(cond_xtx)
    print(f"Condition number order of magnitude: 10^{exponent:.1f}")

# ---------------- Showcase of the hidden layer activation functions ----------------
    z = np.linspace(-5, 5, 100)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=False)
    for ax, (act_y, yd, name) in zip(axes, [
        (sigmoid(z), sigmoid_der(z), "Sigmoid"),
        (ReLU(z), ReLU_der(z), "ReLU"),
        # Added the 'r' prefix here so \alpha isn't swallowed by Python!
        (leaky_ReLU(z, alpha=0.2), leaky_ReLU_der(z, alpha=0.2), r"Leaky ReLU ($\alpha=0.2$)")
    ]):
        ax.plot(z, act_y, label=f"{name} activation")
        ax.plot(z, yd, label=f"{name} derivative")
        ax.set_title(f"{name} activation and derivative")
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.legend()
        
    axes[0].set_ylabel("value")
    axes[0].legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()