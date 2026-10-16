import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit floats, so that results agree with numpy to ~1e-15
import jax.numpy as jnp                    # numpy-compatible functions that JAX can differentiate
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

np.random.seed(2026)   # so that your numbers can be compared with those of other students


# Defining some activation functions
def ReLU(z):
    return jnp.where(z > 0, z, 0.0)


# Derivative of the ReLU function
def ReLU_der(z):
    return jnp.where(z > 0, 1.0, 0.0)


def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))


def mse(predict, target):
    return jnp.mean((predict - target) ** 2)

# ===========================================================================================


def feed_forward_one_layer(W, b, x):
    z = W @ x + b 
    a = sigmoid(z)
    return a


def cost_one_layer(W, b, x, target):
    predict = feed_forward_one_layer(W, b, x)
    return mse(predict, target)


x = np.random.rand(2)
target = np.random.rand(3)

W = np.random.randn(3,2)
b = np.random.randn(3)

jax_one_layer = jax.grad(cost_one_layer, argnums=(0, 1))
W_g, b_g = jax_one_layer(W, b, x, target)
print(W_g, b_g)