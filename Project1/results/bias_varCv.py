# --- BIAS-VARIANCE + CV: the two biggest remaining sections, one script ---
import sys
from pathlib import Path

from sklearn.model_selection import KFold, train_test_split
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

x, y = generate_data(n=100, sigma=0.1, seed=2026)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=0)
degrees = np.arange(1, 16)

# --- CV loop ---
for k in [5, 10]:
    cv_mse = []
    for d in degrees:
        kf = KFold(n_splits=k, shuffle=True, random_state=0)
        scores = []
        for tr, va in kf.split(x_tr):
            Xtr = design_matrix(x_tr[tr], d)
            Xva = design_matrix(x_tr[va], d)
            th  = fit_ols_SVD(Xtr, y_tr[tr])
            scores.append(np.mean((y_tr[va] - Xva @ th) ** 2))
        cv_mse.append(np.mean(scores))
    plt.semilogy(degrees, cv_mse, 'o-', label=f'k={k}')
plt.xlabel('degree'); plt.ylabel('CV MSE'); plt.legend()

# --- Bootstrap bias-variance ---
B = 200
rng = np.random.default_rng(0)
bias_sq, var = [], []
for d in degrees:
    preds = np.zeros((B, len(x_te)))
    for b in range(B):
        idx = rng.integers(0, len(x_tr), len(x_tr))
        Xb  = design_matrix(x_tr[idx], d)
        th  = fit_ols_SVD(Xb, y_tr[idx])
        preds[b] = design_matrix(x_te, d) @ th
    mean_pred = preds.mean(axis=0)
    bias_sq.append(np.mean((y_te - mean_pred) ** 2))
    var.append(np.mean(preds.var(axis=0)))

plt.figure()
plt.semilogy(degrees, bias_sq, 'o-', label=r'Bias$^2$')
plt.semilogy(degrees, var,     's-', label='Variance')
plt.semilogy(degrees, np.array(bias_sq) + np.array(var), '^-', label='Sum')
plt.xlabel('degree'); plt.ylabel('MSE'); plt.legend()
plt.show()