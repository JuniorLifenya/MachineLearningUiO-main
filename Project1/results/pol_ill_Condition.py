import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from src.function_setups import design_matrix, generate_data, runge
from src.fits import fit_ols_SVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- train/test split ---
x, y = generate_data(n=100, sigma=0.1, seed=2026)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=0)

degrees = np.arange(1, 21)
mse_tr, mse_te = [], []
for d in degrees:
    Xtr = design_matrix(x_tr, d)
    Xte = design_matrix(x_te, d)
    th  = fit_ols_SVD(Xtr, y_tr)
    mse_tr.append(mean_squared_error(y_tr, Xtr @ th))
    mse_te.append(mean_squared_error(y_te, Xte @ th))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

# (a) MSE vs degree — the classic U-curve
ax[0].semilogy(degrees, mse_tr, 'o-', color='tab:blue',  label='train MSE')
ax[0].semilogy(degrees, mse_te, 's-', color='tab:red',   label='test MSE')

# mark the test-MSE minimum
i_best = int(np.argmin(mse_te))
d_best = degrees[i_best]
ax[0].axvline(d_best, ls='--', c='k', alpha=0.6,
              label=f'best test $d^*={d_best}$')
ax[0].set_xlabel('polynomial degree')
ax[0].set_ylabel('MSE')
ax[0].set_title('(a) Bias–variance trade-off')
ax[0].legend(); ax[0].grid(alpha=0.3, ls=':')

# (b) example fits at degrees spanning underfit / good / overfit
xx = np.linspace(-1, 1, 400)
show = [3, 8, 13, 18]
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

ax[1].plot(xx, runge(xx), 'k-', lw=2.2, label='Runge (truth)', zorder=5)
ax[1].scatter(x_tr, y_tr, s=20, c='gray', alpha=0.5,
              edgecolor='k', linewidth=0.3, label='train data', zorder=2)
ax[1].scatter(x_te, y_te, s=20, c='white', edgecolor='k',
              linewidth=0.6, label='test data', zorder=2)

for d, c in zip(show, colors):
    Xtr = design_matrix(x_tr, d)
    th  = fit_ols_SVD(Xtr, y_tr)
    yhat = design_matrix(xx, d) @ th
    tag = f'd={d}' + (' (best)' if d == d_best else '')
    ax[1].plot(xx, yhat, '--', color=c, lw=1.8, label=tag)

ax[1].set_ylim(-1.5, 2.5)
ax[1].set_xlabel('x'); ax[1].set_ylabel('y')
ax[1].set_title('(b) Fits at d = 3, 8, 13, 18')
ax[1].legend(fontsize=9, loc='upper center')
ax[1].grid(alpha=0.3, ls=':')

plt.tight_layout()
plt.show()