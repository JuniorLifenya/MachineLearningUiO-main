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
project_root = Path(__file__).parent
sys.path.insert(0,str(project_root))
from function_setup2 import runge, sigmoid, sigmoid_der, ReLU,leaky_ReLU 
from data_aquisition import design_matrix

# ====================================================================
# Main function, weight initialization and data scaling
# ====================================================================

x = np.random.randn(4)

# ---------------- Wight initialization ? ------------------------------

# For a Sigmoid hidden layer, the standard Xavier (Glorot) initialization
# Wl_ij \sim N(0,1/nl_in)...basically zero mean and std = 1/sqrt(n_in)
# For passing signal through many many layers, if the weight are too large
# The activations blow up; and vanish for too small. Choosing this std 
# Keeps the variance roughly constant through the network. With check pdf

# For ReLu, the analogous choice is He initialization: std = sqrt(2/n_in)

def create_layers(network_input_size, layer_output_sizes,seed = None):
    """
        Weights and bias array creation for FFN.

        Wl ~ N(0,1/n_in) Xavier/Glorot normal distr
        std scales as 1/sqrt(n_in), keeps variance of the activations 
        ~ constant through the layers. Prevents vanishing gradients 
        in deep sigmoid networks (Section 8.10)

        bl initialization none. Random weights ~ no random biases
    """
    rng = np.random.default_rng(seed)
    layers = []
    i_size = network_input_size 
    for layer_output_size in layer_output_sizes:
        W = rng.normal(
            loc = 0.0,
            scale = np.sqrt(1.0/i_size), # Xavier for sigmoid
            size = (i_size, layer_output_size),
        )
        b = np.zeros(layer_output_sizes)
        layers.append(W,b)
        i_sizes = layer_output_size
        return layers 


# ---------------- Bias initialization -----------------

bi = np.random.randn(x.size)
# Biases are initialized as constants, s they do not break symmetry.
# The weights are already random (breaking symmetry between neurons)
# So the biases do not need to be random too. Zero bias means the network
# Initially behaves like a pure linear map scaled by the activation
# then training moves the biases to where they need to be.

# Alternative for Relu is small positive constants like 0.01 to prevent
# "Dead neurons" that never fit. But for sigmoid, zeros is standard.

# ---------------- design matrix ----------------------
def design_NN(n,1):
    return x.reshape(-1,1)


# Design/feature matrix setup ? 

# Since OLS/Ridge were linear in the features they needed polyn approx
# Here we do not need that at all
# NN are non linear in the inputs. The composition of many
# Linear layers and nonlinear activations can approx any function
# Recall this as the universal approximation theorem
# NN learns its own features in the hidden layers.

# --- Which activation func for start and final output layer --------
def main_activation(z):
    return sigmoid(z)
# Linear (The identity)--,no activation for final output layer. 
# For regression, the output should be an unbounded real number.
# If sigmoid for the output: Predictions squeezed into (0,1)
# Fine for Runge (whose true values are in (0,1]), but wrong for
# a more general regression. 

# Sigmoid saturates. Near the boundaries of its range, the gradient
# is tiny, and the network learns very slowly exactly where it should
# be most careful (near the extremes of Runge). So really 


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

