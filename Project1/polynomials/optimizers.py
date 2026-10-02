import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from fits import fit_ols_SVD, fit_ridge
from function_setups import generate_data, runge, design_matrix
from fits import fit_ols_SVD,fit_ridge

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ==================================================================
#               Part e)-Gradient Descent, Analytical + AD
# Replace the closed-form θ with an iterative update θ ← θ − η∇C(θ), 
#           using two independent computations 
#       of the gradient that must agree to machine precision.
# ==================================================================

# ------------------- Cost Function First -------------------


def cost_ols(theta,X,y):
    return jnp.mean((y-X @ theta)**2)

def cost_ridge(theta,X,y,lam):
    return cost_ols(theta, X, y) + lam * jnp.dot(theta , theta)

# ------------ Analytical gradients (with 2/nfactor) --------

def grad_ols_analytic(theta,X,y):
    return (2.0 / len(y)) * (X.T @ (X @theta -y))

def grad_ridge_analytic(theta, X, y, lam):
    return grad_ols_analytic(theta, X, y) + 2.0 * lam * theta

# ----------------- AD gradients ----------------------------
grad_ols_ad = jax.grad(cost_ols, argnums=2)
grad_ridge_ad = jax.grad(cost_ridge)

# ----------------- Gradient descent ------------------------
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

# ---------------- Fast verification -------------------------

x,y = generate_data()
degree = 5
X = design_matrix(x,degree, intercept=True)

rng = np.random.default_rng(0)
theta0 = rng.normal(size = X.shape[1])

eta = 0.5 * eta_max_ols(X, y)   # safe: half the theoretical bound
theta_gd, _ = gd(grad_ols_analytic, theta0, n_iter=5000, eta=eta, X=X, y=y)
theta_closed = fit_ols_SVD(X, y)

print(np.max(np.abs(grad_ols_ad(theta,X,y0) - grad_ols_analytic(theta0,X,y))))
print(np.max(np.abs(theta_gd - theta_closed)))   # should be small

# ==================================================================
#           Part f)-ADAGRAD, MOMENTUM, RMSprop, ADAM
#        These optimizers only differ by memory and usage
#           Replace the plain update θ ← θ − η∇C,
#         with a rule that remembers past gradients.
# ==================================================================

def gd_momentum(grad_fn, theta0, n_iter, eta, beta = 0.9, **kw):
    theta = theta0.astype(float).copy()
    v = np.zeros_like(theta)
    for _ in range(n_iter):
        g = np.asarray(grad_fn(theta,**kw))
        v = beta * v + (1- beta)*g # exponential moving avg of grad
        theta -= eta*v

    return theta

def adagrad(grad_fn,theta0, n_iter, eta, eps = 1e-8, **kw):
    theta = theta0.astype(float).copy()
    G = np.zeros_like(theta)
    for _ in range(n_iter):
        g = np.asarray(grad_fn(theta, **kw))
        G += g**2
        theta -=eta*g / (np.sqrt(G) + eps)
    return theta

def rmsprop(grad_fn, theta0, n_iter, eta, beta=0.9, eps = 1e-8, **kw):
    theta = theta0.astype(float).copy()
    s = np.zeros_like(theta)
    for _ in range(n_iter):
        g = np.asarray(grad_fn(theta,**kw))
        s = beta *s + (1-beta)*g**2 # EMA of squared grad
        theta -=eta*g / (np.sqrt(s) + eps)
    return theta

def adam(grad_fn, theta0, n_iter, eta, beta1 =0.9, beta2= 0.999,
        eps = 1e-8, **kw):
    theta = theta0.astype(float).copy()
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    for t in range(1,n_iter + 1):
        g = np.asarray(grad_fn(theta,**kw))
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g**2
        m_hat = m / (1-beta1**t)  # bias correction
        v_hat = v / (1-beta2**t)
        theta -=eta*m_hat / (np.sqrt(v_hat) + eps)

    return theta

# ==================================================================
#                       Part g)-Lasso
#                     Lasso = OLS + λ‖θ‖₁. 
#           Not differentiable at θ = 0, so use a subgradient.
# ==================================================================

def cost_lasso(theta,X,y,lam):
    return jnp.mean((y-X @ theta)**2) + lam * jnp.sum(jnp.abs(theta))

grad_lasso_ad = jax.grad(cost_lasso)

# Use any optimizer from part f), like Adam:
theta_lasso = adam(grad_lasso_ad, theta0, n_iter = 5000,
                   eta = 1e-2, X=X, y=y, lam=1e-3)

# jax returns 0 at θ=0, which is *a* valid subgradient but not the sparsity-inducing one. 
# Contrast with scikit-learn's Lasso, 
# which uses coordinate descent with soft-thresholding and produces exact zeros. 
# Discuss why: proximal operators handle non-differentiable penalties correctly, 
# subgradient descent does not.

def sgd(grad_fn,X,y,theta0,n_epochs, batch_size,eta,
        optimizer ="adam", **kw):
    n = len(y)
    theta = theta0.astype(float).copy()

    # Optimizer State
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    s = np.zeros_like(theta)
    t = 0

    rng = np.random.default_rng(0)

    for epoch in range(n_epochs):
        idx = rng.permutation(n)
        X_sh,y_sh = X[idx], y[idx]
        for start in range(0,n,batch_size):
            Xb = X_sh[start:start + batch_size]
            yb = y_sh[start:start + batch_size]
            g = np.asarray(grad_fn(theta,Xb,yb,**kw))
            t +=1

            if optimizer == "sgd":
                theta -= eta*g
            elif optimizer == "adam":
                m = 0.9*m + 0.1*g
                v = 0.999*v + 0.001*g**2
                m_hat = m / (1-0.9**t)
                v_hat = v / (1-0.999**t)
                theta = theta -eta*m_hat / (np.sqrt(v_hat) + 1e-8)
            # extend for others as needed
    return theta

    