import numpy as np
import matplotlib.pyplot as plt 
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn import datasets
import matplotlib.pyplot as plt
from data_aquisition import design_matrix

# The exercise own starting point
# Runge's function is defined on [-1,1] with 
# Additive Gaussian noise, and pol design matrix

# ============================================================
# 1. THE FUNCTION AND DESIGN MATRIX
# ============================================================

def runge(x):
    """
    Runge's function: f(x) = 1 / (1 + 25 x²).
    """
    return 1.0 / (1.0 + 25.0 * x ** 2)

# ============================================================
# Testable Activation functions four our NN setup later
# ============================================================

def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

def sigmoid_der(z):
    s = sigmoid(z)
    return s* (1.0-s)

def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def ReLU_der(z):
    return jnp.where(z > 0, 1.0, 0.0)

def leaky_ReLU(z, alpha=0.01):
    return jnp.where(z > 0, z, alpha * z)

def leaky_ReLU_der(z, alpha=0.01):
    return jnp.where(z > 0, 1.0, alpha)

# ============================================================
# Fitting Codes 
# ============================================================

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

# =========================================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 100
    sigma = 0.1
    x = np.sort(rng.uniform(-1,1,n))
    y = runge(x) + rng.normal(0, sigma,n)

    xx = np.linspace (-1,1, 400)
    plt.plot(xx, runge(xx), color = "green", label= "Runge function fitting test")
    # Fixed string formatting with 'r'
    plt.scatter(x,y, s = 12, color = "orange",label = r" data, $\sigma = {sigma}$")
    plt.xlabel("x"); plt.ylabel("y");plt.legend(frameon = True)
    plt.show()

    X = design_matrix(x, degree=18, intercept=True)
    cond_xtx = np.linalg.cond(X.T @ X)
    exponent = np.log10(cond_xtx)
    print(f"Condition number order of magnitude: 10^{exponent:.1f}")