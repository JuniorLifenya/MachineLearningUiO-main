import numpy as np

# ==================================================================
#                   Part h)-Stochastic Grad Descent
#                     Same optimizers as before,
#               but each step uses a random mini-bach instead
#                   No full training set used here.
# ==================================================================

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

# mini-batch size (16, 32, 128, full), number of epochs, learning-rate schedule (constant vs decaying).
# Discuss: smaller batches are noisier but escape sharp minima; 
# larger batches are more stable but slower per epoch. 
# Cost per epoch is higher for SGD because of the Python loop, so SGD only wins for very large n.

# ==================================================================
#                   Part i)-Final Model Selection
#                     Use the CV setup from part d) 
#               Pick (degree, lam) for OLS, Ridge, and Lasso,
#                       Compare the three here
# ==================================================================    

# src/selection.py

import numpy as np
from resampling import cv_score_degree


def cv_full_grid(x, y, degrees, lambdas, k=5, seed=42):
    """
    CV MSE for OLS, Ridge, Lasso over the (degree, lambda) grid.
    """
    D, L = len(degrees), len(lambdas)
    grid = {
        "OLS":   np.zeros(D),
        "Ridge": np.zeros((D, L)),
        "Lasso": np.zeros((D, L)),
    }

    for i, d in enumerate(degrees):
        grid["OLS"][i] = cv_score_degree(x, y, d, k=k, lam=0.0, seed=seed)
        for j, lam in enumerate(lambdas):
            grid["Ridge"][i, j] = cv_score_degree(x, y, d, k=k, lam=lam, seed=seed)
            # For Lasso, cv_score_degree currently only supports OLS/Ridge.
            # Extend it with a `method` argument (see below).
    return grid


def best_config(grid, degrees, lambdas):
    """Return (method, degree, lambda, cv_mse) for the global minimum."""
    i = int(np.argmin(grid["OLS"]))
    best = ("OLS", degrees[i], None, grid["OLS"][i])

    for name in ("Ridge",):
        i, j = np.unravel_index(np.argmin(grid[name]), grid[name].shape)
        if grid[name][i, j] < best[3]:
            best = (name, degrees[i], lambdas[j], grid[name][i, j])
    return best

def cv_score_degree(x, y, degree, k=5, lam=0.0, method="ols", seed=42):
    ...
    for tr_idx, va_idx in kf.split(X_all):
        ...
        if degree == 0:
            y_pred = np.full(len(y_va), y_tr.mean())
            fold_mses.append(np.mean((y_va - y_pred) ** 2))
            continue

        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_va = scaler.transform(X_va)

        y_mean = y_tr.mean()
        y_tr_c = y_tr - y_mean

        if method == "ols":
            theta = ols_fit(X_tr, y_tr_c)
        elif method == "ridge":
            theta = ridge_fit(X_tr, y_tr_c, lam)
        elif method == "lasso":
            theta = fit_lasso_gd(X_tr, y_tr_c, lam)
        else:
            raise ValueError(f"Unknown method: {method}")

        y_pred = X_va @ theta + y_mean
        fold_mses.append(np.mean((y_va - y_pred) ** 2))
    return float(np.mean(fold_mses))

def fit_lasso_gd(X, y, lam, n_iter=2000, eta=1e-2):
    """
    Lasso via gradient descent on the AD gradient.
    Subgradient method — theta does NOT become exactly zero.
    """
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    def cost(theta, X, y, lam):
        return jnp.mean((y - X @ theta) ** 2) + lam * jnp.sum(jnp.abs(theta))

    grad = jax.grad(cost)
    theta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        g = np.asarray(grad(theta, X, y, lam))
        theta = theta - eta * g
    return theta

# Cell 10 — Part i)
degrees = list(range(0, 16))
lambdas = np.logspace(-6, 0, 20)

grid = cv_full_grid(x, y, degrees, lambdas, k=5)
method, d_best, lam_best, mse_best = best_config(grid, degrees, lambdas)

print(f"Best model:  {method}")
print(f"Best degree: {d_best}")
print(f"Best lambda: {lam_best}")
print(f"CV MSE:      {mse_best:.4e}")

plot_part_i(grid, degrees, lambdas)   # 4-panel figure