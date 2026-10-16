import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that results agree with numpy to ~1e-15
import jax.numpy as jnp                    # numpy-compatible functions that JAX can differentiate
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# Defining some activation functions
def ReLU(z):
    return jnp.where(z > 0, z, 0.0)


def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time."""
    e_z = jnp.exp(z - jnp.max(z))
    return e_z / jnp.sum(e_z)


# ================================ Exercise 1 =================================================================

np.random.seed(2024)
x = np.random.randn(2)      # shape (2,)      — one input, two features
W1 = np.random.randn(4, 2)  # shape (4, 2)    — 4 output neurons, 2 inputs each

b1 = np.random.randn(4) # Each output neuron has their won bias 

z1 = W1 @ x  + b1 # Has shape (4,)
a1 = ReLU(z1) # Becomes the input for the next laer, with dimensions (4,)

# ================================ Exercise 2 =================================================================

w2 = np.random.randn(8,4) # But why is the output now 8 ? What is the pattern for this increase hm
b2 = np.random.randn(8) # Following the (n_out,n_in) format 

z2 = w2 @ a1 + b2
a2 = ReLU(z2)

# ================================ Exercise 3 =================================================================
# Here we Generalize for many many layers 

def many_layers(network_input_size,layer_output_sizes):
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size,i_size)
        b = np.random.randn(layer_output_size)
        layers.append((W,b))
        i_size =layer_output_size

    return layers 

def feed_forward_all_relu(layers,input):
    a = input
    for W, b in layers:
        z = W @ a + b
        a = ReLU(z)
    return a

input_size = 8 
layer_output_sizes = [10,16,6,2]
x = np.random.randn(input_size)
layers = many_layers(input_size,layer_output_sizes)
predict = feed_forward_all_relu(layers,x)
print(predict)

# ================================ Exercise 4 =================================================================

def feed_forward(input, layers, activation_funcs):
    a = input
    for (W,b), activation_func in zip(layers,activation_funcs):
        z = W @ a + b 
        a = ReLU(z)
    return a 

network_input_size = ...
layer_output_sizes = [...]
activation_funcs = [ReLU, ReLU, sigmoid]
layers = ...

x = np.random.randn(network_input_size)
feed_forward(x, layers, activation_funcs)

# ================================ Exercise 5 =================================================================
