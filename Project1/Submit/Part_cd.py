import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from Part_ab import generate_data, runge, design_matrix


# ------------------------------------------------------------
# SVD fitters
# ------------------------------------------------------------

