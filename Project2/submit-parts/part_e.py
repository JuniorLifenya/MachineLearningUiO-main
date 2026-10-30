import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from function_setup2 import runge, sigmoid, sigmoid_der
from data_aquisition import design_matrix
from part_a import C_mse, dC_dypred_batch, L1_penalty_grad, L2_penalty_grad
from part_b import (
    I_identity, dI_dz, raw_x, create_layers, backpropagation_batch,
    adam_step, predict,
)

with np.load("mnist.npz") as data:
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]  # untouched until the end
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

#split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

x_train = x_train.reshape(len(x_train), -1) / 255.0
x_val = x_val.reshape(len(x_val), -1) / 255.0
x_test = x_test.reshape(len(x_test), -1) / 255.0
