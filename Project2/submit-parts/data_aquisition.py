import numpy as np
import pandas as pd 
import sys 
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
project_root = Path(__file__).parent
sys.path.insert(0,str(project_root))
from function_setup2 import runge, design_matrix, generate_data

# ============================================================
# Design Matrix with polynomial setup
# ============================================================

def design_matrix(x, degree, intercept=True):
    """
    Polynomial design matrix.

    Columns are [1, x, x², ..., x^degree] if intercept=True,
    or [x, x², ..., x^degree] if intercept=False.

    """
    x = np.asarray(x).ravel()
    start = 0 if intercept else 1
    return np.vstack([x ** p for p in range(start, degree + 1)]).T

# ============================================================
# Data Generation
# ============================================================

# Runge data (same as project 1)
rng = np.random.default_rng(2026)
n = 100
sigma = 0.1
x = np.sort(rng.uniform(-1,1,n))
y = runge(x) + rng.normal(0, sigma,n)



def generate_data(n=100, sigma=0.1, seed=2026, x_min=-1.0, x_max=1.0):
    """
        Sample Runge's function with additive Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(x_min, x_max, n))
    y = runge(x) + rng.normal(0.0, sigma, n)
    return x, y

xx = np.linspace (-1,1, 400)
plt.plot(xx, runge(xx), color = "green", label= "Runge function fitting test")
plt.scatter(x,y, s = 12, color = "orange",label = r" data, $\sigma = {sigma}$")
plt.xlabel("x"); plt.ylabel("y");plt.legend(frameon = True)
plt.show()
