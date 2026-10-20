# Reusing our gradient machinery from project 1,
# We need the following functions and gradients
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys 
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0,str(project_root))
from function_setup import runge, design_matrix, generate_data

# ================== 1 MSE functions ==============================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_L1(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) + abs

def mse_L2(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) + abs**2

# ================== 2 Binary cross entropy (log loss) ============

def cross_entropy()
    