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

theta1 = np.linspace(-2, 2, 400)
theta2 = np.linspace(-2, 2, 400)
T1, T2 = np.meshgrid(theta1, theta2)

# a toy cost landscape: ellipses centred at (0.7, 1.2)
ctr = np.array([0.7, 1.2])
C = (T1 - ctr[0])**2 / 1.5 + (T2 - ctr[1])**2 / 0.8

fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))

for a, (constraint_fn, name) in zip(ax, [
    (lambda t1, t2: np.zeros_like(t1) > 1,        'OLS (no constraint)'),
    (lambda t1, t2: t1**2 + t2**2 <= 1.0,         r'Ridge ($\ell_2$)'),
    (lambda t1, t2: np.abs(t1) + np.abs(t2) <= 1.2, r'Lasso ($\ell_1$)'),
]):
    a.contour(T1, T2, C, levels=15, cmap='Greys', linewidths=0.7)
    a.contourf(T1, T2, constraint_fn(T1, T2).astype(float),
               levels=[0.5, 1.5], colors=['tab:blue'], alpha=0.25)
    a.axhline(0, c='k', lw=0.5); a.axvline(0, c='k', lw=0.5)
    a.set_xlim(-2, 2); a.set_ylim(-2, 2); a.set_aspect('equal')
    a.set_title(name)
    a.set_xlabel(r'$\theta_1$'); a.set_ylabel(r'$\theta_2$')
plt.tight_layout()
plt.show()