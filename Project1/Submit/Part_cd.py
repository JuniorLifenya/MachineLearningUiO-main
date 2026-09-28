import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from Part_ab import generate_data, runge, design_matrix
from function_setup import fit_ols_SVD,fit_ridge


# ==================================================================
# Training and Test MSE
# ==================================================================

def training_MSE(y,y_tilde,pol_deg):
    for i in range(pol_deg):
        Mean_training = np.mean(y)
        Mean_Test = np.mean(y_tilde)
    


