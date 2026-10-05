import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso as SklearnLasso

from Part_ab import generate_data, design_matrix
from function_setup import fit_ols_SVD

# ==================================================================
#           Part e) Cost Functions & Gradients (Analytical & AD)
# ==================================================================

def cost_ols(theta, X, y):
    return jnp.mean((y - X @ theta) ** 2)

def cost_ridge(theta, X, y, lam):
    return cost_ols(theta, X, y) + lam * jnp.dot(theta, theta)

def grad_ols_analytic(theta, X, y):
    return (2.0 / len(y)) * (X.T @ (X @ theta - y))

def grad_ridge_analytic(theta, X, y, lam):
    return grad_ols_analytic(theta, X, y) + 2.0 * lam * theta

# FIXED: argnums=0 differentiates with respect to theta
grad_ols_ad = jax.grad(cost_ols, argnums=0)
grad_ridge_ad = jax.grad(cost_ridge, argnums=0)

def eta_max_ols(X, y):
    H = (2.0 / len(y)) * (X.T @ X)
    return 2.0 / np.linalg.eigvalsh(H).max()

# --- Gradient Descent ---
def gd(grad_fn, theta0, n_iter=5000, eta=1e-3, **kwargs):
    theta = theta0.astype(float).copy()
    hist = [theta.copy()]
    for _ in range(n_iter):
        g = np.asarray(grad_fn(theta, **kwargs))
        theta -= eta * g
        hist.append(theta.copy())
    return theta, np.array(hist)


# ==================================================================
#          Part f) - Optimizers with History Tracking
# ==================================================================

def gd_momentum(grad_fn, theta0, n_iter, eta, beta=0.9, **kw):
    theta = theta0.astype(float).copy()
    v = np.zeros_like(theta)
    hist = []
    for _ in range(n_iter):
        g = np.asarray(grad_fn(theta, **kw))
        v = beta * v + (1 - beta) * g
        theta -= eta * v
        hist.append(cost_ols(theta, kw['X'], kw['y'])) # Track MSE
    return theta, hist

def adagrad(grad_fn, theta0, n_iter, eta, eps=1e-8, **kw):
    theta = theta0.astype(float).copy()
    G = np.zeros_like(theta)
    hist = []
    for _ in range(n_iter):
        g = np.asarray(grad_fn(theta, **kw))
        G += g**2
        theta -= eta * g / (np.sqrt(G) + eps)
        hist.append(cost_ols(theta, kw['X'], kw['y']))
    return theta, hist

def rmsprop(grad_fn, theta0, n_iter, eta, beta=0.9, eps=1e-8, **kw):
    theta = theta0.astype(float).copy()
    s = np.zeros_like(theta)
    hist = []
    for _ in range(n_iter):
        g = np.asarray(grad_fn(theta, **kw))
        s = beta * s + (1 - beta) * g**2
        theta -= eta * g / (np.sqrt(s) + eps)
        hist.append(cost_ols(theta, kw['X'], kw['y']))
    return theta, hist

def adam(grad_fn, theta0, n_iter, eta, beta1=0.9, beta2=0.999, eps=1e-8, **kw):
    theta = theta0.astype(float).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    hist = []
    for t in range(1, n_iter + 1):
        g = np.asarray(grad_fn(theta, **kw))
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        theta -= eta * m_hat / (np.sqrt(v_hat) + eps)
        hist.append(cost_ols(theta, kw['X'], kw['y']))
    return theta, hist

# ==================================================================
#          Plotting Optimizer Convergence
# ==================================================================
# --- Data Setup & Initialization ---
x, y = generate_data()
degree = 5
X = design_matrix(x, degree, intercept=True)

rng = np.random.default_rng(0)
theta0 = rng.normal(size=X.shape[1])

# Define shared hyperparameters
iters = 1000
learning_rate = 0.05

# Define shared hyperparameters
iters = 1000
learning_rate = 0.05

# Run optimizers and capture history
_, hist_mom = gd_momentum(grad_ols_analytic, theta0, iters, learning_rate, X=X, y=y)
_, hist_ada = adagrad(grad_ols_analytic, theta0, iters, learning_rate, X=X, y=y)
_, hist_rms = rmsprop(grad_ols_analytic, theta0, iters, learning_rate, X=X, y=y)
_, hist_adam = adam(grad_ols_analytic, theta0, iters, learning_rate, X=X, y=y)

# Plot the learning curves
plt.figure(figsize=(10, 6))
plt.plot(hist_mom, label="Momentum")
plt.plot(hist_ada, label="AdaGrad")
plt.plot(hist_rms, label="RMSprop")
plt.plot(hist_adam, label="Adam")

plt.yscale("log")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE) - Log Scale")
plt.title(f"Convergence Comparison (eta = {learning_rate})")
plt.legend()
plt.grid(True)

if __name__ == "__main__":
    x, y = generate_data()
    degree = 5
    X_unscaled = design_matrix(x, degree, intercept=False)
    
    # Scale X and center y
    scaler = StandardScaler().fit(X_unscaled)
    X = scaler.transform(X_unscaled)
    y_mean = y.mean()
    y_c = y - y_mean

    rng = np.random.default_rng(0)
    theta0 = rng.normal(size=X.shape[1])

    # 1. Gradient Agreement Test
    grad_analytic = grad_ols_analytic(theta0, X, y_c)
    grad_ad = np.array(grad_ols_ad(theta0, X, y_c))
    max_grad_diff = np.max(np.abs(grad_analytic - grad_ad))
    
    print(f"Max Gradient Difference (Analytic vs JAX AD): {max_grad_diff:.2e}")

    # 2. GD vs Closed-Form Solution
    eta = 0.5 * eta_max_ols(X, y_c)
    theta_gd, _ = gd(grad_ols_analytic, theta0, n_iter=10000, eta=eta, X=X, y=y_c)
    theta_closed = fit_ols_SVD(X, y_c)
    max_theta_diff = np.max(np.abs(theta_gd - theta_closed))
    
    print(f"Max Difference GD vs Closed-Form (OLS SVD)   : {max_theta_diff:.2e}")

    # 3. Part g: Subgradient Descent vs. Coordinate Descent (Lasso)
    def cost_lasso(theta, X, y, lam):
        return cost_ols(theta, X, y) + lam * jnp.sum(jnp.abs(theta))

    grad_lasso_ad = jax.grad(cost_lasso, argnums=0)
    lam = 1e-2
    
    # Subgradient via Adam
    theta_subgrad, _ = adam(grad_lasso_ad, theta0, n_iter=10000, eta=1e-2, X=X, y=y_c, lam=lam)
    
    # True Lasso via Scikit-Learn (Coordinate Descent)
    sk_lasso = SklearnLasso(alpha=lam, fit_intercept=False, max_iter=10000)
    sk_lasso.fit(X, y_c)
    theta_sklearn = sk_lasso.coef_

    print("\n--- LASSO WEIGHT COMPARISON ---")
    print(f"Adam (Subgradient) Weights : {np.round(theta_subgrad, 5)}")
    print(f"Sklearn (Coordinate Desc)  : {np.round(theta_sklearn, 5)}")

# --- 4. Table I: Iterations to reach convergence (theta distance < 10^-4) ---
    print("\n--- TABLE I: OPTIMIZER ITERATIONS TO CONVERGE ---")
    theta_target = fit_ols_SVD(X, y_c)
    tol = 1e-4
    max_iters = 50000
    eta_table = 1e-2

    # 1. Plain GD
    th = theta0.copy()
    iters_gd = max_iters
    for i in range(max_iters):
        g = grad_ols_analytic(th, X, y_c)
        th -= eta_table * g
        if np.max(np.abs(th - theta_target)) < tol:
            iters_gd = i + 1
            break
    print(f"Plain GD                       : {iters_gd}")

    # 2. Momentum
    th = theta0.copy()
    v = np.zeros_like(th)
    beta = 0.9
    iters_mom = max_iters
    for i in range(max_iters):
        g = grad_ols_analytic(th, X, y_c)
        v = beta * v + (1 - beta) * g
        th -= eta_table * v
        if np.max(np.abs(th - theta_target)) < tol:
            iters_mom = i + 1
            break
    print(f"Momentum (beta = 0.9)          : {iters_mom}")

    # 3. AdaGrad
    th = theta0.copy()
    G = np.zeros_like(th)
    eps = 1e-8
    iters_ada = max_iters
    for i in range(max_iters):
        g = grad_ols_analytic(th, X, y_c)
        G += g**2
        th -= eta_table * g / (np.sqrt(G) + eps)
        if np.max(np.abs(th - theta_target)) < tol:
            iters_ada = i + 1
            break
    print(f"AdaGrad                        : {iters_ada}")

    # 4. RMSprop
    th = theta0.copy()
    s = np.zeros_like(th)
    beta = 0.9
    eps = 1e-8
    iters_rms = max_iters
    for i in range(max_iters):
        g = grad_ols_analytic(th, X, y_c)
        s = beta * s + (1 - beta) * g**2
        th -= eta_table * g / (np.sqrt(s) + eps)
        if np.max(np.abs(th - theta_target)) < tol:
            iters_rms = i + 1
            break
    print(f"RMSprop (beta = 0.9)           : {iters_rms}")

    # 5. Adam
    th = theta0.copy()
    m = np.zeros_like(th)
    v = np.zeros_like(th)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    iters_adam = max_iters
    for t_step in range(1, max_iters + 1):
        g = grad_ols_analytic(th, X, y_c)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t_step)
        v_hat = v / (1 - beta2**t_step)
        th -= eta_table * m_hat / (np.sqrt(v_hat) + eps)
        if np.max(np.abs(th - theta_target)) < tol:
            iters_adam = t_step
            break
    print(f"Adam (beta1=0.9, beta2=0.999)  : {iters_adam}")