import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys 
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0,str(project_root))
from src.function_setups import design_matrix, runge, generate_data
from src.fits import fit_ols_SVD, fit_ridge, predict, standardize_polynomials



def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    