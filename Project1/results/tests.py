import sys 
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0,str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from polynomials.optimizers import gd
from polynomials.metrics import ols_fit, predict, fit_ols_SVD
from src.function_setups import runge, design_matrix, generate_data


# ================= First we test SVD-based OLS fit on a simple example =================
x,y = generate_data(n=100, sigma=0.1, seed=2026)
degree = 5 #Can also be changed to be for 8-15
X = design_matrix(x, degree, intercept=True)
xx = np.linspace(-1, 1, 400)
Xx = design_matrix(xx, degree, intercept=True)

# ------------- Compare two solutions -------------------
theta_ols_svd = fit_ols_SVD(X, y, rcond=1e-12) # "analytical" (closed form)
theta_gd = gd(X, y, n_iter=5000, eta=1e-3) # iterative (gradient descent)

fig,ax = plt.subplots(2,2, figsize=(10,6))

# ------------ Plot the two fits -------------------
ax[0,0].scatter(xx,runge(xx), color="green", label="Runge function")
ax[0,0].scatter(x,y, color="red", s=12, label="data")
ax[0,0].plot(xx, Xx @ theta_ols_svd, 'b--', label=r'SVD $\theta$')
ax[0,0].plot(xx, Xx @ theta_gd, 'm:', label=r'GD $\theta$')
ax[0,0].legend();ax[0,0].set_title(f"Polynomial degree {degree} fit")

plt.tight_layout();plt.show()