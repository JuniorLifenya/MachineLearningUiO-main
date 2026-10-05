import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Ensure JAX uses double precision
jax.config.update("jax_enable_x64", True)

from Part_ab import generate_data, design_matrix
from function_setup import fit_ols_SVD, fit_ridge

# ==================================================================
#                  Part h) Stochastic Grad Descent
# ==================================================================

def cost_fn(theta, X, y):
    """Computes the full-batch Mean Squared Error cost."""
    return np.mean((X @ theta - y) ** 2)

def mse_grad(theta, X, y):
    """Computes the gradient of the MSE with respect to theta."""
    return 2.0 / len(y) * X.T @ (X @ theta - y)

def sgd_with_history(grad_fn, X, y, theta0, n_epochs, batch_size, eta, optimizer="adam", **kw):
    """SGD with history tracking for plotting training cost over epochs."""
    n = len(y)
    theta = theta0.astype(float).copy()

    # Optimizer State (Adam)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    t = 0
    rng = np.random.default_rng(42)
    
    cost_history = []
    b_size = n if batch_size == 'n' or batch_size >= n else batch_size

    for epoch in range(n_epochs):
        cost_history.append(cost_fn(theta, X, y))
        
        idx = rng.permutation(n)
        X_sh, y_sh = X[idx], y[idx]
        
        for start in range(0, n, b_size):
            Xb = X_sh[start:start + b_size]
            yb = y_sh[start:start + b_size]
            g = np.asarray(grad_fn(theta, Xb, yb, **kw))
            t += 1

            if optimizer == "sgd":
                theta -= eta * g
            elif optimizer == "adam":
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g**2
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                theta = theta - eta * m_hat / (np.sqrt(v_hat) + 1e-8)
                
    return theta, cost_history

# ==================================================================
#                  Part i) Final Model Selection (CV)
# ==================================================================    

# JIT compile the cost and gradient ONCE outside the loop for massive speedup
def lasso_cost(theta, X, y, lam):
    return jnp.mean((y - jnp.dot(X, theta)) ** 2) + lam * jnp.sum(jnp.abs(theta))

lasso_grad = jax.jit(jax.grad(lasso_cost, argnums=0))

def fit_lasso_gd(X, y, lam, n_iter=2000, eta=1e-2):
    """
    Lasso via gradient descent on the AD gradient.
    Subgradient method — theta does NOT become exactly zero.
    """
    theta = jnp.zeros(X.shape[1]) # Keep as jnp array during updates
    for _ in range(n_iter):
        g = lasso_grad(theta, X, y, lam)
        theta = theta - eta * g
    return np.asarray(theta) # Convert back to standard numpy at the very end

def cv_score_degree(x, y, degree, k=5, lam=0.0, method="ols", seed=42):
    """Evaluates a specific degree and lambda using K-Fold CV."""
    X_all = design_matrix(x, degree) if degree > 0 else np.zeros((len(x), 1))
    
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_mses = []
    
    for tr_idx, va_idx in kf.split(X_all):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
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
            theta = fit_ols_SVD(X_tr, y_tr_c) 
        elif method == "ridge":
            theta = fit_ridge(X_tr, y_tr_c, lam)
        elif method == "lasso":
            theta = fit_lasso_gd(X_tr, y_tr_c, lam)
        else:
            raise ValueError(f"Unknown method: {method}")

        y_pred = X_va @ theta + y_mean
        fold_mses.append(np.mean((y_va - y_pred) ** 2))
        
    return float(np.mean(fold_mses))

def cv_full_grid(x, y, degrees, lambdas, k=5, seed=42):
    """CV MSE for OLS, Ridge, Lasso over the (degree, lambda) grid."""
    D, L = len(degrees), len(lambdas)
    grid = {
        "OLS":   np.zeros(D),
        "Ridge": np.zeros((D, L)),
        "Lasso": np.zeros((D, L)),
    }

    for i, d in enumerate(degrees):
        grid["OLS"][i] = cv_score_degree(x, y, d, k=k, lam=0.0, method="ols", seed=seed)
        for j, lam in enumerate(lambdas):
            grid["Ridge"][i, j] = cv_score_degree(x, y, d, k=k, lam=lam, method="ridge", seed=seed)
            grid["Lasso"][i, j] = cv_score_degree(x, y, d, k=k, lam=lam, method="lasso", seed=seed)
            
    return grid

def best_config(grid, degrees, lambdas):
    """Return (method, degree, lambda, cv_mse) for the global minimum."""
    i = int(np.argmin(grid["OLS"]))
    best = ("OLS", degrees[i], None, grid["OLS"][i])

    # Now checks both Ridge and Lasso for the global minimum
    for name in ("Ridge", "Lasso"):
        i, j = np.unravel_index(np.argmin(grid[name]), grid[name].shape)
        if grid[name][i, j] < best[3]:
            best = (name, degrees[i], lambdas[j], grid[name][i, j])
            
    return best

# ==================================================================
#                  Main Execution Block
# ==================================================================

if __name__ == "__main__":
    x, y = generate_data(n=100, sigma=0.1, seed=2026)
    
    # -----------------------------------------------------------
    # Run Grid Search (Part i)
    # -----------------------------------------------------------
    print("--- Running Cross-Validation Grid Search ---")
    degrees = list(range(0, 16))
    lambdas = np.logspace(-6, 0, 20)

    grid = cv_full_grid(x, y, degrees, lambdas, k=5)
    method, d_best, lam_best, mse_best = best_config(grid, degrees, lambdas)

    print(f"Best model:  {method}")
    print(f"Best degree: {d_best}")
    print(f"Best lambda: {lam_best}")
    print(f"CV MSE:      {mse_best:.4e}")
    
    # -----------------------------------------------------------
    # Run SGD Plotting (Part h)
    # -----------------------------------------------------------
    print("\n--- Running SGD Optimization Plot ---")
    test_degree = 3 
    X_poly = design_matrix(x, test_degree)
    theta0 = np.zeros(X_poly.shape[1])

    n_epochs = 500
    eta = 1e-2
    n = len(y)
    batch_sizes = [8, 32, 128, n]

    plt.figure(figsize=(9, 5))

    for bs in batch_sizes:
        label_str = f"Batch size B = {bs}" if bs != n else f"Batch size B = n ({n})"
        _, history = sgd_with_history(
            mse_grad, X_poly, y, theta0, 
            n_epochs=n_epochs, 
            batch_size=bs, 
            eta=eta, 
            optimizer="adam"
        )
        plt.plot(history, label=label_str, alpha=0.85, linewidth=1.5)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Training Cost (MSE)", fontsize=12)
    plt.title(f"Training Cost vs. Epoch (Adam Optimizer, eta = {eta})", fontsize=13)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig("sgd_training_cost.pdf", bbox_inches="tight")
    plt.show()