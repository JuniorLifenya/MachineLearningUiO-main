# In this part I am goint to reuse some codes I have made earlier for my 
# Other finance project really
import numpy as np 
import matplotlib.pyplot as plt
from Part_ab import generate_data
from Part_ab import runge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression


# ===========================================================================================
#                   Data Generation and analysis
# ===========================================================================================

# --- First we make a plot similar to 2.11 from Hastie, Tibishirani and Friedman ----
# The plot visually showcases models prediction error as a function sof complexity
# We model complexity as...I am unsure here

data = generate_data()
model = LinearRegression()
y = runge(x=x)
X_sklearn = ...I am unsure here what to use really
y_tilde = model.predict(X_sklearn)
MSE = mean_squared_error(y,y_tilde)