import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys 
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0,str(project_root))
from function_setup2 import runge, sigmoid, sigmoid_der, ReLU,leaky_ReLU 
from data_aquisition import design_matrix
from part_a import C_mse, dC_dypred_batch, dBC_dypred

# =========================================================================
# PART 1) Main function, weight initialization and data scaling
# =========================================================================

# ---------------- Wights and biases initialization  ----------------------

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
            scale = np.sqrt(1.0/i_size), # Xavier for sigmoid, 
            # Compare to week 41 np.random.randn(i_size,layer_output_size)
            # Which used std = 1. Difference between trainable NN or not !
            size = (i_size, layer_output_size),
        )
        b = np.zeros(layer_output_size)
        layers.append((W,b))
        i_size = layer_output_size # Batched convention in project description
    return layers 

# Biases are initialized as constants, as they do not break symmetry.
# The weights are already random (breaking symmetry between neurons)
# So the biases do not need to be random too. Zero bias means the network
# Initially behaves like a pure linear map scaled by the activation
# then training moves the biases to where they need to be.

# Alternative for Relu is small positive constants like 0.01 to prevent
# "Dead neurons" that never fit. But for sigmoid, zeros is standard.

# ---------------- design matrix ----------------------------------------

# NN input: raw x, reshaped to a column. No polynomial features.
# The network learns its own features in the hidden layers.
def raw_x(x):
    return np.asarray(x).reshape(-1, 1)

# Quick Plot to confirm the data and compare
if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 100
    sigma = 0.1
    x = np.sort(rng.uniform(-1,1,n))
    y = runge(x) + rng.normal(0.0, sigma,n)

    xx = np.linspace(-1,1,400)
    plt.plot(xx,runge(xx), color = "green", label = "Runge function again")
    plt.scatter(x,y, s = 12, color = "orange",
                label = rf"data, $\sigma = {sigma}$")
    plt.xlabel("x");plt.ylabel("y")
    plt.legend();plt.show()

    X = design_matrix(x, degree = 18, intercept = True)
    print(f"Condition number of X^TX: 10^{np.log10(np.linalg.cond(X.T @ X)):.1f}")


# Since OLS/Ridge were linear in the features they needed polyn approx
# Here we do not need that at all
# NN are non linear in the inputs. The composition of many
# Linear layers and nonlinear activations can approx any function
# Recall this as the universal approximation theorem
# NN learns its own features in the hidden layers.

# ---------- Which activation func for start and final output layer -------------

A = sigmoid(z)

# Linear (The identity)--,no activation for final output layer. 
def I_identity(z):
    return z
def dI_dz(z):
    return np.ones_like(z)
# For regression, the output should be an unbounded real number.
# If sigmoid for the output: Predictions squeezed into (0,1)
# Fine for Runge (whose true values are in (0,1]), but wrong for
# a more general regression. 

# Sigmoid saturates. Near the boundaries of its range, the gradient
# is tiny, and the network learns very slowly exactly where it should
# be most careful (near the extremes of Runge). So really 


# =========================================================================
# PART 2) Train the Network and Compare OLS from project 1
# =========================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Standardize input (fit on training only)
scaler_x = StandardScaler().fit(X_train)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

# Reshape y for the batched backprop (one output per sample)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Use 1-dim Runge function. Use same data sets which gave the best MSE
# Use polynomial order from project 1 with the best result.
# Compare results with NN with on and two hidden layers
# Use 50 and 100 hidden nodes, respectively
# Comment results, give critical discussion of OLS results 
# from project 1 code and NN here
# Make analysis of the learning rates employed to find the optimal MSE
# score. Test both SG and RMSprop and Adam and GD with different learning rates  

# =========================================================================

# ----- First FF from week 42 ----------------------
def feed_forward_saver_batch(input, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a

# --------- Also BackProp batch From week 42 --------

def backpropagation_batch(inputs, layers, activation_funcs, targets, activation_ders, cost_der):
    layer_inputs, zs, predict = feed_forward_saver_batch(inputs, layers, activation_funcs)
    layer_grads = [() for layer in layers]
    dC_dZ = None

    for i in reversed(range(len(layers))):
        A_prev = layer_inputs[i]
        Z = zs[i]
        actv_der = activation_ders[i]

        if i == len(layers) - 1:
            dC_dZ = cost_der(predict, targets)
        else:
            W_next = layers[i + 1][0]
            dC_dZ = dC_dZ @ W_next.T

        dC_dZ = dC_dZ * actv_der(Z)
        dC_dW = A_prev.T @ dC_dZ
        dC_db = dC_dZ.sum(axis=0)

        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads


# ----------- Optimization algos/methods -----------------------------

def sgd_step(layers,grads,gamma):
    return [(W-gamma * dW,b-gamma * db) for (W,b),(dW,b) in zip(layers,grads)]

def rmsprop_step(layers,grads,state,gamma,beta=.9,eps=1e-8):
    new_state = []
    new_layers = []
    for (W,b), (dW,db), (sW,sb) in zip(layers,grads,state):
        sW = beta*sW + (1-beta)*dW**2
        sb = beta*sb + (1-beta)*db**2
        new_layers.append((W-gamma*dW / (np.sqrt(sW) + eps),
                           b -gamma*db / (np.sqrt(sb) + eps)))
        new_state.append((sW,sb))
    return new_layers, new_state

def adam_step(layers,grads,state,gamma,t,
              beta1 = .9, beta2 = .999, eps = 1e-8):
    new_state = []
    new_layers = []
    for (W,b), (dW,db), (mW,vW,mb, vb) in zip (layers,grads,state):
        mW = beta1 * mW + (1-beta1)*dW
        vW = beta2 * vW + (1-beta2)*dW**2
        mb = beta1*mb + (1-beta1)*db
        vb = beta2*vb + (1-beta2)*db**2

        mW_hat = mW/(1-beta1**t)
        vW_hat = vW/(1-beta2**t)
        mb_hat = mb/(1-beta1**t)
        vb_hat = vb/(1-beta2**t)

        new_layers.append((W-gamma*mW_hat / (np.sqrt(vW_hat) + eps),
                           b - gamma*mb_hat / (np.sqrt(vb_hat) + eps)))
        new_state.append((mW,vW,mb,vb))
    return new_layers,new_state

def train_network(X_train,y_train,layers,
                  actv_funcs,actv_ders,optimizer = "adam",
                  gamma = 1e-3, n_epochs = 500,
                  batch_size = 32, seed = 42):
    n = len(y_train)
    rng = np.random_default_rng(2026)

    # Initialize optimizer state 
    if optimizer == "rmsprop":
        state = [(np.zeros_like(W),np.zeros_like(b)) for W,b in layers]
    elif optimizer == "adam":
        state = [(np.zeros_like(W), np.zeros_like(W),
                  np.zeros_like(b), np.zeros_like(b)) for W,b in layers]
    else:
        state = None

    t = 0
    for epoch in range(n_epochs):
        idx = rng.permutation(n)
        for start in range (0,n,batch_size):
            Xb = X_train[idx[start:start + batch_size]]
            yb = y_train[idx[start:start + batch_size]]
            grads = backpropagation_batch(Xb,layers,actv_funcs,yb,
                    actv_ders,dC_dypred_batch)
        if optimizer == "sgd":
            layers = sgd_step(layers,grads,gamma)
        elif optimizer == "rmsprop":
            layers,state = rmsprop_step(layers,grads,state,gamma)
        elif optimizer == "adam":
            t +=1
            layers, state = adam_step(layers,grads,state,gamma,t)
    return layers 

def predict(X,layers, actv_funcs):
    _, _, a = feed_forward_saver_batch(X,layers,actv_funcs)
    return a

# ============================================================
# PART 3 — Driver (runs when you execute the file)
# ============================================================



