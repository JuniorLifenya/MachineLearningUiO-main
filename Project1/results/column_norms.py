import sys 
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0,str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from src.function_setups import design_matrix, generate_data
from src.function_setups import runge
from src.fits import fit_ols_SVD
import importlib, polynomials.optimizers
importlib.reload(polynomials.optimizers)

x, y = generate_data(n=100, sigma=0.1, seed=2026)
xx   = np.linspace(-1, 1, 400)

degrees = np.arange(1, 21)
dist, kappa_XtX = [], []
theta_dir_all, theta_svd_all = [], []

for d in degrees:
    Xd = design_matrix(x, d)                 # NO standardisation — that's the point
    theta_dir = np.linalg.solve(Xd.T @ Xd, Xd.T @ y)   # formal normal equations
    theta_svd = fit_ols_SVD(Xd, y)                     # SVD
    theta_dir_all.append(theta_dir)
    theta_svd_all.append(theta_svd)
    dist.append(np.max(np.abs(theta_dir - theta_svd)))
    kappa_XtX.append(np.linalg.cond(Xd.T @ Xd))

# print for your own sanity
for d, k, dv in zip(degrees, kappa_XtX, dist):
    print(f"deg {d:2d}:  κ(XᵀX) = {k:.2e}   ‖θ_dir − θ_SVD‖∞ = {dv:.2e}")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# (a) Diagnostic: divergence AND its cause
ax[0].semilogy(degrees, dist,      'ko-',  label=r'$\|\theta_{\rm formal}-\theta_{\rm SVD}\|_\infty$')
ax[0].semilogy(degrees, kappa_XtX, 'rs-',  label=r'$\kappa(\mathbf{X}^T\mathbf{X})$')
ax[0].axhline(1e-15, ls=':', c='gray', label=r'$\epsilon_{\rm machine}$')
ax[0].axhline(1e16,  ls=':', c='r',    label=r'$1/\epsilon_{\rm machine}$')
ax[0].axvline(18, ls='--', c='k', alpha=0.5)
ax[0].set_xlabel('polynomial degree')
ax[0].set_title('Divergence and its cause')
ax[0].legend(fontsize=8)

# (b) Fits at the failure degree
d_hi = degrees[np.argmax(np.array(dist) > 0.1)]      # ~18
Xd  = design_matrix(x, d_hi); Xxd = design_matrix(xx, d_hi)
i   = list(degrees).index(d_hi)
ax[1].plot(xx, runge(xx), 'g-', lw=2, label='Runge')
ax[1].scatter(x, y, s=10, c='orange', label='data')
ax[1].plot(xx, Xxd @ theta_dir_all[i], 'b--', lw=1.5, label=r'$\theta_{\rm formal}$ (direct solve)')
ax[1].plot(xx, Xxd @ theta_svd_all[i], 'r:',  lw=1.5, label=r'$\theta_{\rm SVD}$')
ax[1].set_ylim(-10, 10)
ax[1].legend()
ax[1].set_title(f'degree = {d_hi}: formal solver fails')

plt.tight_layout()
plt.show()