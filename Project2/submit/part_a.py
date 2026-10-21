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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred have different shapes")
    
    return np.mean((y_true - y_pred) ** 2)

def l1_penalty(parameters):
    return sum(np.sum(np.abs(p)) for p in parameters)

def l2_penalty(parameters):
    return sum(np.sum(np.asarray(p)**2 for p in parameters))

def mse_l1(y_true,y_pred,parameters,lam):
    return mse(y_true,y_pred) + lam * l1_penalty(parameters)

def mse_l2(y_true,y_pred,parameters,lam):
    return mse(y_true,y_pred) + lam * l2_penalty(parameters)

# ================== 2 Binary cross entropy (log loss) ============

def cross_entropy()
    