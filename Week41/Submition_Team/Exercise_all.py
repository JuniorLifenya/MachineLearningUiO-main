import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

def softmax(z):
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def softmax_vec(z):
    e_z = jnp.exp(z - jnp.max(z))
    return e_z / jnp.sum(e_z)


# ============ Exercise 1 ============
np.random.seed(2024)
x = np.random.randn(2)
W1 = np.random.randn(4, 2)
b1 = np.random.randn(4)

z1 = W1 @ x + b1
a1 = ReLU(z1)

sol1 = np.array([0.60610368, 4.0076268, 0.0, 0.56469864])
print("Ex 1:", np.allclose(a1, sol1))


# ============ Exercise 2 ============
W2 = np.random.randn(8, 4)
b2 = np.random.randn(8)

z2 = W2 @ a1 + b2
a2 = ReLU(z2)

print("Ex 2:", np.allclose(np.exp(len(a2)), 2980.9579870417283))


# ============ Exercise 3 ============
def many_layers(network_input_size, layer_output_sizes):
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))
        i_size = layer_output_size
    return layers


def feed_forward_all_relu(layers, input):
    a = input
    for W, b in layers:
        z = W @ a + b
        a = ReLU(z)
    return a


input_size = 8
layer_output_sizes = [10, 16, 6, 2]
x = np.random.randn(input_size)
layers = many_layers(input_size, layer_output_sizes)
predict = feed_forward_all_relu(layers, x)
print("Ex 3 output shape:", predict.shape)


# ============ Exercise 4 ============
def feed_forward(input, layers, activation_funcs):
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = W @ a + b
        a = activation_func(z)          # use the passed activation
    return a


network_input_size = 5
layer_output_sizes = [8, 6, 4]
activation_funcs = [ReLU, ReLU, sigmoid]
layers = many_layers(network_input_size, layer_output_sizes)

x = np.random.randn(network_input_size)
output = feed_forward(x, layers, activation_funcs)
print("Ex 4 output:", output)


# ============ Exercise 5 ============
def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(i_size, layer_output_size)   # (n_in, n_out)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))
        i_size = layer_output_size                        # update i_size
    return layers


def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs                                             
    for (W, b), activation_func in zip(layers, activation_funcs):   
        z = a @ W + b                                      # Fixed a @ W, not W @ a
        a = activation_func(z)
    return a


inputs = np.random.randn(1000, 4)
network_input_size = 4
layer_output_sizes = [12, 10, 3]
activation_funcs = [ReLU, ReLU, softmax]
layers = create_layers_batch(network_input_size, layer_output_sizes)

output = feed_forward_batch(inputs, layers, activation_funcs)
print("Ex 5 output shape:", output.shape)   # expect (1000, 3)


# ============ Exercise 6 ============
iris = datasets.load_iris()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names,
    loc="lower right", title="Classes",
)
plt.show()                                                

inputs = iris.data
targets = np.zeros((len(iris.data), 3))
for i, t in enumerate(iris.target):
    targets[i, t] = 1


def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)
    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)


network_input_size = 4
layer_output_sizes = [8, 3]
activation_funcs = [sigmoid, softmax]
layers = create_layers_batch(network_input_size, layer_output_sizes)
predictions = feed_forward_batch(inputs, layers, activation_funcs)
print("Ex 6 accuracy:", accuracy(predictions, targets))