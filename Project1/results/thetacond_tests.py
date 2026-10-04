import sys 
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0,str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from src.fits import fit_ols_SVD
from src.function_setups import runge, design_matrix, generate_data
import jax
from polynomials.optimizers import eta_max_ols, cost_ols,cost_ridge, grad_ols_analytic, grad_ridge_analytic, gd
import importlib, polynomials.optimizers
importlib.reload(polynomials.optimizers)

# =================================================================
# (1) First we test SVD-based OLS fit on a simple example 
# =================================================================

x,y = generate_data(n=100, sigma=0.1, seed=2026)
degree = 5 # For higher degree, the design matrix becomes rank-deficient and the normal equations fail.
X = design_matrix(x, degree, intercept=True)
xx = np.linspace(-1, 1, 400)
Xx = design_matrix(xx, degree, intercept=True)

# ------------- Compare two solutions -------------------
rng = np.random.default_rng(0)
theta0 = rng.normal(size=X.shape[1])


degrees = np.arange(1, 16)
dist    = []
theta_dir_all = []
theta_svd_all = []

for d in degrees:
    Xd = design_matrix(x, d)
    theta_dir = np.linalg.solve(Xd.T @ Xd, Xd.T @ y)   # formal normal eqs
    theta_svd = fit_ols_SVD(Xd, y)                     # SVD
    theta_dir_all.append(theta_dir)
    theta_svd_all.append(theta_svd)
    dist.append(np.max(np.abs(theta_dir - theta_svd)))

fig, ax = plt.subplots(1, 3, figsize=(14, 4))

# (a) degree 5: they agree
d_lo = 5
Xd  = design_matrix(x,  d_lo); Xxd = design_matrix(xx, d_lo)
i   = list(degrees).index(d_lo)
ax[0].plot(xx, runge(xx), 'g-', label='Runge')
ax[0].scatter(x, y, s=10, c='orange', label='data')
ax[0].plot(xx, Xxd @ theta_dir_all[i], 'b--', label=r'$\theta_{\rm formal}$')
ax[0].plot(xx, Xxd @ theta_svd_all[i], 'r:',  label=r'$\theta_{\rm SVD}$')
ax[0].legend(); ax[0].set_title(f'degree = {d_lo}: identical')

# (b) divergence vs degree
ax[1].semilogy(degrees, dist, 'ko-')
ax[1].set_xlabel('polynomial degree')
ax[1].set_ylabel(r'$\|\theta_{\rm formal}-\theta_{\rm SVD}\|_\infty$')
ax[1].set_title('Numerical divergence')

# (c) degree 15: formal solver breaks
d_hi = 15
Xd  = design_matrix(x,  d_hi); Xxd = design_matrix(xx, d_hi)
i   = list(degrees).index(d_hi)
ax[2].plot(xx, runge(xx), 'g-', label='Runge')
ax[2].scatter(x, y, s=10, c='orange', label='data')
ax[2].plot(xx, Xxd @ theta_dir_all[i], 'b--', label=r'$\theta_{\rm formal}$')
ax[2].plot(xx, Xxd @ theta_svd_all[i], 'r:',  label=r'$\theta_{\rm SVD}$')
ax[2].set_ylim(-3, 3)
ax[2].legend(); ax[2].set_title(f'degree = {d_hi}: formal fails')

plt.tight_layout()
plt.show()