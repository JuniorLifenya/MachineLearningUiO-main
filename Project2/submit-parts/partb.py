import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys 
from pathlib import Path
project_root = Path(__file__).parent.parent()
sys.path.insert(0,str(project_root))
from function_setup import runge, design_matrix, generate_data
from part_a import sigmoid

# ====================================================================
# Main function, weight initialization and data scaling
# ====================================================================

def main_activation(z):
    return sigmoid(z)

# Weight Initialization of the weights with normal distr
# Width of the distr should depend on the # of layer inputs
# Bias initialization ? Which activation func for final output layer
# Design/feature matrix setup ? 
# Hint: have to represent a polynomial approx like project 1 ?

# ====================================================================

# Train the Network and Compare OLS from project 1
# Use 1-dim Runge function. Use same data sets wich gave the best MSE
# Use polynomial order from project 1 with the best result.
# Compare results with NN with on and two hidden layers
# Use 50 and 100 hidden nodes, respectively
# Comment results, give critical discussion of OLS results 
# from project 1 code and NN here
# Make analysis of the learning rates employed to find the optimal MSE
# score. Test both SG and RMSprop and Adam and GD with different learning rates  
# ====================================================================

