#============================ Case 1: My own gradient descent for OLS ===============================
# 1) Scale: standardise the columns 
import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.linear_model import Ridge

X = np.column_stack([x, x**2])

# Standardize features (zero mean, unit variance for each feature)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1  # safeguard to avoid division by zero for constant features
X_norm = (X - X_mean) / X_std

# Center the target to zero mean (optional, to simplify intercept handling)
y_mean = y.mean()
y_centered = y-y.mean()
