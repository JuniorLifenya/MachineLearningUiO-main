import sys 
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0,str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from src.fits import ols_fit, fit_ols_SVD, fit_ridge, predict
from src.function_setups import runge, design_matrix, generate_data
import jax
import jax.numpy as jnp
from polynomials.optimizers import eta_max_ols, cost_ols,cost_ridge, grad_ols_analytic, grad_ridge_analytic, gd, gd_momentum, adagrad, rmsprop, adam
import importlib, polynomials.optimizers
importlib.reload(polynomials.optimizers)

# =================================================================
# (1) First we test SVD-based OLS fit on a simple example 
# =================================================================

x,y = generate_data(n=100, sigma=0.1, seed=2026)
degree = 5 #Can also be changed to be for 8-15
X = design_matrix(x, degree, intercept=True)
xx = np.linspace(-1, 1, 400)
Xx = design_matrix(xx, degree, intercept=True)

# ------------- Compare two solutions -------------------
rng = np.random.default_rng(0)
theta0 = rng.normal(size=X.shape[1])

eta = 0.5 * eta_max_ols(X, 100)
eta_explode = 1.5 * eta_max_ols(X, 100)

theta_gd, hist_gd = gd(grad_ols_analytic, theta0,
                       n_iter=5000, eta=eta, X=X, y=y)
theta_svd = fit_ols_SVD(X, y)

# =================================================================
# (2) GRADIENT CHECK:  analytic vs JAX AD, at the same theta
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
# CONVERGENCE / STABILITY PLOT
# =================================================================
# trajectories for the stability panel
_, hist_safe    = gd(grad_ols_analytic, theta0, n_iter=30, eta=eta,         X=X, y=y)
_, hist_explode = gd(grad_ols_analytic, theta0, n_iter=30, eta=eta_explode, X=X, y=y)

fig, ax = plt.subplots(2, 2, figsize=(11, 7))

# (a) fits — do GD and closed-form look the same?

ax[0,0].plot(xx, runge(xx), 'g-', label='Runge')
ax[0,0].scatter(x, y, s=12, c='orange', label='data')
ax[0,0].plot(xx, Xx @ theta_svd, 'b--', label='SVD')
ax[0,0].plot(xx, Xx @ theta_gd,  'r:',  label='GD')
ax[0,0].legend(); ax[0,0].set_title(f'degree={degree}')

# (b) convergence of GD toward SVD
dist = [np.linalg.norm(t - theta_svd, ord=np.inf) for t in hist_gd]
ax[1,0].semilogy(dist)
ax[1,0].axhline(1e-4, ls='--', c='k')
ax[1,0].set_xlabel('iteration')
ax[1,0].set_ylabel(r'$\|\theta_k-\theta_{\rm SVD}\|_\infty$')
ax[1,0].set_title('GD convergence')

# (c) plot exploding step size:  GD diverges if η > η_max

dist_safe    = [np.linalg.norm(t - theta_svd, ord=np.inf) for t in hist_safe]
dist_explode = [np.linalg.norm(t - theta_svd, ord=np.inf) for t in hist_explode]

ax[0,1].semilogy(dist_safe,    'b-',  label=r'$\eta = 0.5\,\eta_{\max}$')
ax[0,1].semilogy(dist_explode, 'r--', label=r'$\eta = 1.5\,\eta_{\max}$')
ax[0,1].axhline(1e-4, ls=':', c='k', label=r'$10^{-4}$ tol')
ax[0,1].set_xlabel('iteration')
ax[0,1].set_ylabel(r'$\|\theta_k - \theta_{\rm SVD}\|_\infty$')
ax[0,1].legend()
ax[0,1].set_title('Convergence vs divergence')


# (d) Plots of GD marching toward SVD solution
iters = {
  "Iteration: 5": 5,
  "Iteration: 50": 50,
  "Iteration: 500": 500,
  "Iteration: 5000": 5000
}
for label, k in iters.items():
    ax[1,1].plot(xx, Xx @ hist_gd[k], alpha=0.8,
            label=fr'GD, {k} iters')
ax[1,1].plot(xx, Xx @ theta_svd, 'k--', lw=2, label=f'SVD {label}')

plt.tight_layout()
plt.savefig("test_gd.png", dpi=140)
plt.legend(loc='center', bbox_to_anchor=(0.5, 0.3), ncol=2)
plt.show()

