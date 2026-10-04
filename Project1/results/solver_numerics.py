import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from src.function_setups import design_matrix, generate_data, runge
from src.fits import fit_ols_SVD

x, y = generate_data(n=100, sigma=0.1, seed=2026)

d = 18                                    # the degree where it breaks
X  = design_matrix(x,  d)
th_dir = np.linalg.solve(X.T @ X, X.T @ y)
th_svd = fit_ols_SVD(X, y)

xx_in  = np.linspace(-1, 1, 400)          # interior (training range)
xx_out = np.linspace(-2, 2, 800)          # extended (extrapolation)

y_dir_in  = design_matrix(xx_in,  d) @ th_dir
y_svd_in  = design_matrix(xx_in,  d) @ th_svd
y_dir_out = design_matrix(xx_out, d) @ th_dir
y_svd_out = design_matrix(xx_out, d) @ th_svd

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# (a) Coefficients side by side — THE key panel
j = np.arange(d + 1)
w = 0.4
rel_diff = np.abs(th_dir - th_svd) / (np.abs(th_svd) + 1e-30)

ax[0,0].bar(j - 0.2, np.abs(th_svd),  0.4, label='SVD',    color='tab:red')
ax[0,0].bar(j + 0.2, np.abs(th_dir),  0.4, label='Formal', color='tab:blue')
ax[0,0].set_yscale('log')
ax[0,0].set_xlabel('coefficient index $j$')
ax[0,0].set_ylabel(r'$|\theta_j|$')
ax[0,0].set_title(
    rf'Coefficient magnitudes at $d={d}$'
    '\n'
    rf'(max relative difference: {np.max(rel_diff[np.abs(th_svd) > 1e-10]):.1e})'
)
ax[0,0].legend(fontsize=9)
ax[0,0].grid(alpha=0.3, ls=':', axis='y')

# (b) Prediction difference on the training interval
diff_in = np.abs(y_dir_in - y_svd_in)
ax[0,1].semilogy(xx_in, diff_in + 1e-18, 'k-')
ax[0,1].axhline(1e-15, ls=':', c='gray', lw=0.8)
ax[0,1].set_xlabel('$x$')
ax[0,1].set_ylabel(r'$|y_{\rm dir}(x) - y_{\rm SVD}(x)|$')
max_diff_in = np.max(diff_in)

ax[0,1].set_title(
    rf'Prediction difference on $[-1,1]$'
    '\n'
    rf'$\max |p_{{\rm dir}}-p_{{\rm SVD}}|={max_diff_in:.1e}$'
)
ax[0,1].set_ylim(1e-18, 1e2)
ax[0,1].grid(alpha=0.3, ls=':')

# (c) Fits on training range — visually indistinguishable
ax[1,0].plot(xx_in, runge(xx_in), 'g-', lw=2.0, label='Runge')
ax[1,0].scatter(x, y, s=14, c='orange', edgecolor='k',
                linewidth=0.3, label='data', zorder=3)
ax[1,0].plot(xx_in, y_dir_in, 'b--', lw=1.6, label='Formal')
ax[1,0].plot(xx_in, y_svd_in, 'r:',  lw=1.6, label='SVD')
ax[1,0].set_ylim(-0.5, 1.5)
ax[1,0].set_xlabel('$x$'); ax[1,0].set_ylabel('$y$')
ax[1,0].set_title('(c) Both solvers: same fit on training range')
ax[1,0].legend(fontsize=9); ax[1,0].grid(alpha=0.3, ls=':')

# (d) Extrapolation — the practical consequence
ax[1,1].plot(xx_out, runge(xx_out), 'g-', lw=2.0, label='Runge (truth)')
ax[1,1].axvspan(-1, 1, color='gray', alpha=0.15, label='training range')
ax[1,1].plot(xx_out, y_dir_out, 'b--', lw=1.6, label='Formal')
ax[1,1].plot(xx_out, y_svd_out, 'r:',  lw=1.6, label='SVD')
ax[1,1].set_ylim(-50, 50)
ax[1,1].set_xlabel('$x$'); ax[1,1].set_ylabel('$y$')
ax[1,1].set_title('(d) Outside training range: formal solve explodes')
ax[1,1].legend(fontsize=9); ax[1,1].grid(alpha=0.3, ls=':')

plt.tight_layout()
plt.savefig("solver_numerics.png", dpi=150, bbox_inches='tight')
plt.show()