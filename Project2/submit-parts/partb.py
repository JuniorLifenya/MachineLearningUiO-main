import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from function_setup2 import runge, sigmoid, sigmoid_der, ReLU, ReLU_der, leaky_ReLU, leaky_ReLU_der
from data_aquisition import design_matrix
from part_a import C_mse, dC_dypred_batch


def create_layers(network_input_size, layer_output_sizes, seed=None):
    rng = np.random.default_rng(seed)
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = rng.normal(0.0, np.sqrt(1.0 / i_size),
                       size=(i_size, layer_output_size))
        b = np.zeros(layer_output_size)
        layers.append((W, b))
        i_size = layer_output_size
    return layers


def I_identity(z):
    return z


def dI_dz(z):
    return np.ones_like(z)


def raw_x(x):
    return np.asarray(x).reshape(-1, 1)


def feed_forward_saver_batch(inputs, layers, activation_funcs):
    layer_inputs, zs = [], []
    a = inputs
    for (W, b), activation in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W + b
        a = activation(z)
        zs.append(z)
    return layer_inputs, zs, a


def backpropagation_batch(inputs, layers, activation_funcs, targets,
                          activation_ders, cost_der):
    layer_inputs, zs, predict = feed_forward_saver_batch(
        inputs, layers, activation_funcs
    )
    layer_grads = [None] * len(layers)
    dC_dZ = None

    for i in reversed(range(len(layers))):
        A_prev = layer_inputs[i]
        Z = zs[i]
        actv_der = activation_ders[i]

        if i == len(layers) - 1:
            dC_dZ = cost_der(targets, predict)
        else:
            W_next = layers[i + 1][0]
            dC_dZ = dC_dZ @ W_next.T

        dC_dZ = dC_dZ * actv_der(Z)
        dC_dW = A_prev.T @ dC_dZ
        dC_db = dC_dZ.sum(axis=0)
        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads


def sgd_step(layers, grads, gamma):
    return [(W - gamma * dW, b - gamma * db)
            for (W, b), (dW, db) in zip(layers, grads)]


def rmsprop_step(layers, grads, state, gamma, beta=0.9, eps=1e-8):
    new_state, new_layers = [], []
    for (W, b), (dW, db), (sW, sb) in zip(layers, grads, state):
        sW = beta * sW + (1 - beta) * dW ** 2
        sb = beta * sb + (1 - beta) * db ** 2
        new_layers.append((
            W - gamma * dW / (np.sqrt(sW) + eps),
            b - gamma * db / (np.sqrt(sb) + eps),
        ))
        new_state.append((sW, sb))
    return new_layers, new_state


def adam_step(layers, grads, state, gamma, t,
              beta1=0.9, beta2=0.999, eps=1e-8):
    new_state, new_layers = [], []
    for (W, b), (dW, db), (mW, vW, mb, vb) in zip(layers, grads, state):
        mW = beta1 * mW + (1 - beta1) * dW
        vW = beta2 * vW + (1 - beta2) * dW ** 2
        mb = beta1 * mb + (1 - beta1) * db
        vb = beta2 * vb + (1 - beta2) * db ** 2

        mW_hat = mW / (1 - beta1 ** t)
        vW_hat = vW / (1 - beta2 ** t)
        mb_hat = mb / (1 - beta1 ** t)
        vb_hat = vb / (1 - beta2 ** t)

        new_layers.append((
            W - gamma * mW_hat / (np.sqrt(vW_hat) + eps),
            b - gamma * mb_hat / (np.sqrt(vb_hat) + eps),
        ))
        new_state.append((mW, vW, mb, vb))
    return new_layers, new_state


def train_network(X_train, y_train, layers, actv_funcs, actv_ders,
                  optimizer="adam", gamma=1e-3, n_epochs=500,
                  batch_size=32, seed=42):
    n = len(y_train)
    rng = np.random.default_rng(seed)

    if optimizer == "rmsprop":
        state = [(np.zeros_like(W), np.zeros_like(b)) for W, b in layers]
    elif optimizer == "adam":
        state = [(np.zeros_like(W), np.zeros_like(W),
                  np.zeros_like(b), np.zeros_like(b)) for W, b in layers]
    else:
        state = None

    t = 0
    for epoch in range(n_epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            Xb = X_train[idx[start:start + batch_size]]
            yb = y_train[idx[start:start + batch_size]]
            grads = backpropagation_batch(
                Xb, layers, actv_funcs, yb, actv_ders, dC_dypred_batch
            )
            if optimizer == "sgd":
                layers = sgd_step(layers, grads, gamma)
            elif optimizer == "rmsprop":
                layers, state = rmsprop_step(layers, grads, state, gamma)
            elif optimizer == "adam":
                t += 1
                layers, state = adam_step(layers, grads, state, gamma, t)

    return layers


def predict(X, layers, actv_funcs):
    _, _, a = feed_forward_saver_batch(X, layers, actv_funcs)
    return a


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 100
    sigma = 0.1
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    y = runge(x) + rng.normal(0.0, sigma, n)

    X_nn = raw_x(x)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_nn, y.reshape(-1, 1), test_size=0.3, random_state=42
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    deg_best = 10
    X_poly = design_matrix(x, degree=deg_best, intercept=True)
    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(
        X_poly, y, test_size=0.3, random_state=42
    )
    theta_ols = np.linalg.pinv(Xp_tr.T @ Xp_tr) @ Xp_tr.T @ yp_tr
    mse_ols = C_mse(yp_te, Xp_te @ theta_ols)
    print(f"OLS (degree {deg_best}): test MSE = {mse_ols:.4e}")

    actv_funcs = [sigmoid, sigmoid, I_identity]
    actv_ders = [sigmoid_der, sigmoid_der, dI_dz]

    configs = [
        ("1 hidden, 50 nodes", [50, 1]),
        ("1 hidden, 100 nodes", [100, 1]),
        ("2 hidden, 50/50", [50, 50, 1]),
        ("2 hidden, 100/100", [100, 100, 1]),
    ]

    print(f"\n{'Config':<25} {'Optimizer':<10} {'gamma':<10} {'Test MSE':<12}")
    print("-" * 60)
    for name, out_sizes in configs:
        for opt, gamma in [("sgd", 1e-2), ("rmsprop", 1e-3), ("adam", 1e-3)]:
            layers = create_layers(1, out_sizes, seed=42)
            acts = actv_funcs[:len(out_sizes)]
            ders = actv_ders[:len(out_sizes)]
            layers = train_network(
                X_tr, y_tr, layers, acts, ders,
                optimizer=opt, gamma=gamma,
                n_epochs=500, batch_size=32,
            )
            test_mse = C_mse(y_te, predict(X_te, layers, acts))
            print(f"{name:<25} {opt:<10} {gamma:<10.0e} {test_mse:<12.4e}")
