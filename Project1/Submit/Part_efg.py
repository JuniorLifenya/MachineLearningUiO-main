import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from Part_ab import generate_data, runge, design_matrix
from function_setup import fit_ols_SVD,fit_ridge
from Part_cd import cv_score_degree, bootstrap_bias_variance


# ==================================================================
# ------------------- Cost Function First ----------------
# ==================================================================

def cost_ols(X,y,theta):
    return jnp.mean((y-X @ theta)**2)

def cost_ridge(theta,X,y,lam):
    return cost_ols(theta, X, y) + lam * jnp.dot(theta , theta)

# ==================================================================
# ------------ Analytical gradients (with 2/nfactor) --------
# ==================================================================

def grad_ols_analytic(theta,X,y):
    return (2.0 / len(y)) * (X.T @ (X @theta -y))

def grad_ridge_analytic(theta, X, y, lam):
    return grad_ols_analytic(theta, X, y) + 2.0 * lam * theta

# ----------------- AD gradients -------------------
grad_ols_ad = jax.grad(cost_ols, argnums=2)
grad_ridge_ad = jax.grad(cost_ridge)

# ----------------- Gradient descent ---------------
def gd(grad_fn, theta0, n_iter = 5000, eta = 1e-3, **kwargs):
    """
        Plain gradient descent.
        grad_fn(theta, **kwargs) -> gradient vector.
        Returns (theta_final, theta_history).
    """
    theta = theta0.astype(float).copy()
    hist = [theta.copy()]
    for _ in range(n_iter):
        g = np.asarray(grad_fn(theta, **kwargs))
        theta = theta - eta*g
        hist.append(theta.copy())

    return theta, np.array(hist)

def eta_max_ols(X, y):
    H = (2.0 / len(y)) * (X.T @ X)
    return 2.0 / np.linalg.eigvalsh(H).max()

# ---------------- Fast verification ----------------
X,y = generate_data()
rng = np.random.default_rng(0)
theta0 = rng.normal(size = X.shape[1])

eta = 0.5 * eta_max_ols(X, y)   # safe: half the theoretical bound
theta_gd, _ = gd(grad_ols_analytic, theta0, n_iter=5000, eta=eta, X=X, y=y)
theta_closed = fit_ols_SVD(X, y)

print(np.max(np.abs(grad_ols_ad(theta0,X,y) - grad_ols_analytic(theta0,X,y))))
print(np.max(np.abs(theta_gd - theta_closed)))   # should be small