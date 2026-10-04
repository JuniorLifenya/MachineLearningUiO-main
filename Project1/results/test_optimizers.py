import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0,str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from src.function_setups import generate_data, runge, design_matrix
from src.fits import fit_ols_SVD, fit_ridge, predict
from polynomials.optimizers import (cost_ols, cost_ridge, cost_lasso,
                        grad_ols_analytic, grad_ridge_analytic,
                        gd, gd_momentum, adagrad, rmsprop, adam,
                        eta_max_ols)

# ---- data ----
x, y = generate_data(n=100, sigma=0.1, seed=2026)
degree = 5
X = design_matrix(x, degree, intercept=True)

rng = np.random.default_rng(0)
theta0 = rng.normal(size=X.shape[1])

# =================================================================
# (1) GRADIENT CHECK:  analytic vs JAX AD, at the same theta
# =================================================================
g_an_ols   = grad_ols_analytic(theta0, X, y)
g_ad_ols   = jax.grad(cost_ols, argnums=0)(theta0, X, y)

lam = 1e-2
g_an_ridge = grad_ridge_analytic(theta0, X, y, lam)
g_ad_ridge = jax.grad(cost_ridge, argnums=0)(theta0, X, y, lam)

print(f"OLS   |∇_an − ∇_ad|_∞ = {np.max(np.abs(g_an_ols - g_ad_ols)):.2e}")
print(f"Ridge |∇_an − ∇_ad|_∞ = {np.max(np.abs(g_an_ridge - g_ad_ridge)):.2e}")
# Both should print ~1e-15 to 1e-16 (machine precision).

# =================================================================
# (2) CLOSED-FORM CHECK:  GD must converge to fit_ols_SVD
# =================================================================
eta = 0.5 * eta_max_ols(X, 100)
theta_gd, hist_gd = gd(grad_ols_analytic, theta0,
                       n_iter=5000, eta=eta, X=X, y=y)
theta_svd = fit_ols_SVD(X, y)

print(f"GD vs SVD  |Δθ|_∞ = {np.max(np.abs(theta_gd - theta_svd)):.2e}")
# Should be small (< 1e-3 at degree 5).

# =================================================================
# (3) CONVERGENCE / STABILITY PLOT
# =================================================================
fig, ax = plt.subplots(2, 2, figsize=(11, 7))

# (a) fits — do GD and closed-form look the same?
xx  = np.linspace(-1, 1, 400)
Xx  = design_matrix(xx, degree, intercept=True)
ax[0,0].plot(xx, runge(xx), 'g-', label='Runge')
ax[0,0].scatter(x, y, s=12, c='orange', label='data')
ax[0,0].plot(xx, Xx @ theta_svd, 'b--', label='SVD')
ax[0,0].plot(xx, Xx @ theta_gd,  'r:',  label='GD')
ax[0,0].legend(); ax[0,0].set_title(f'degree={degree}')

# (b) normal-equation residual for SVD fit  (should be ~1e-12)
resid = X.T @ (X @ theta_svd - y)
ax[0,1].bar(np.arange(degree+1), np.abs(resid))
ax[0,1].set_yscale('log')
ax[0,1].set_title(r'$|X^T(X\theta_{\rm SVD}-y)|$')

# (c) convergence of GD toward SVD
dist = [np.linalg.norm(t - theta_svd, ord=np.inf) for t in hist_gd]
ax[1,0].semilogy(dist)
ax[1,0].axhline(1e-4, ls='--', c='k')
ax[1,0].set_xlabel('iteration'); ax[1,0].set_ylabel(r'$\|\theta_k-\theta_{\rm SVD}\|_\infty$')
ax[1,0].set_title('GD convergence')

# (d) cost along the GD trajectory
cost = [cost_ols(t, X, y) for t in hist_gd]
ax[1,1].semilogy(cost); ax[1,1].set_xlabel('iteration')
ax[1,1].set_ylabel(r'$C(\theta_k)$'); ax[1,1].set_title('cost')

plt.tight_layout(); plt.savefig("test_gd.png", dpi=140)

# =================================================================
# (4) OPTIMIZER RACE  (must all share the same return interface)
# =================================================================
tol   = 1e-4
n_it  = 2000
opts = {
    "GD":       gd,
    "Momentum": gd_momentum,
    "AdaGrad":  adagrad,
    "RMSprop":  rmsprop,
    "Adam":     adam,
}
iters_to_tol = {}
for name, fn in opts.items():
    _, h = fn(grad_ols_analytic, theta0, n_iter=n_it, eta=eta, X=X, y=y)
    d = [np.max(np.abs(t - theta_svd)) for t in h]
    hit = next((i for i, v in enumerate(d) if v < tol), None)
    iters_to_tol[name] = hit
    print(f"{name:>9s}: {hit} iters to tol")

# =================================================================
# (5) LASSO vs SKLEARN  (sparsity claim)
# =================================================================
from sklearn.linear_model import Lasso
lam_l = 1e-3
theta_lasso_ours = adam(jax.grad(cost_lasso, argnums=0), theta0,
                        n_iter=5000, eta=1e-2, X=X, y=y, lam=lam_l)[0]

sk = Lasso(alpha=lam_l, fit_intercept=False, max_iter=10000).fit(X, y)

print(f"ours   nonzero: {np.sum(np.abs(theta_lasso_ours) > 1e-6)}")
print(f"sklearn nonzero: {np.sum(np.abs(sk.coef_) > 1e-6)}")