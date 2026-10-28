# verify_grad.py
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)          # match numpy float64

from function_setup2 import sigmoid, sigmoid_der
from part_a          import C_mse, dC_dypred_batch
from partb           import (create_layers, backpropagation_batch,
                             I_identity, dI_dz,predict)

# ---- 2 hidden (sigmoid) + linear output ----
seed   = 42
layers = create_layers(1, [5, 5, 1], seed=seed)
acts   = [sigmoid,      sigmoid,      I_identity]
ders   = [sigmoid_der,  sigmoid_der,  dI_dz]

# ---- fixed batch ----
rng = np.random.default_rng(0)
B   = 8
X   = rng.normal(size=(B, 1))
Y   = rng.normal(size=(B, 1))

# ---- analytic gradients ----
grads_analytic = backpropagation_batch(
    X, layers, acts, Y, ders, dC_dypred_batch)

# ---- JAX gradients ----
params_jax = [(jnp.asarray(W), jnp.asarray(b)) for W, b in layers]
X_j, Y_j   = jnp.asarray(X), jnp.asarray(Y)

def forward(params, X):
    a = X
    for i, (W, b) in enumerate(params):
        z = a @ W + b
        a = jax.nn.sigmoid(z) if i < len(params) - 1 else z   # last is linear
    return a

def cost(params, X, Y):
    return jnp.mean((forward(params, X) - Y) ** 2)

# sanity: the two costs must agree too
print("C_mse =", C_mse(Y, np.asarray(forward(params_jax, X_j))))
print("JAX   =", float(cost(params_jax, X_j, Y_j)))

grads_jax = jax.grad(cost)(params_jax, X_j, Y_j)

def cost_only(pred, Y):
    return jnp.mean((pred - Y) ** 2)

pred_np   = predict(X, layers, acts)          # same forward, numpy
pred_j    = jnp.asarray(pred_np)
cost_der_jax    = jax.grad(cost_only)(pred_j, Y_j)       # (B,1)
cost_der_manual = dC_dypred_batch(Y, pred_np)            # (B,1)?

print("shapes:", cost_der_manual.shape, cost_der_jax.shape)
ratio = np.asarray(cost_der_manual) / np.asarray(cost_der_jax)
print("ratio (should all be 1.0):")
print(ratio.ravel())

# ---- compare ----
max_err = 0.0
for i, ((dW_a, db_a), (dW_j, db_j)) in enumerate(zip(grads_analytic, grads_jax)):
    eW = np.max(np.abs(dW_a - np.asarray(dW_j)))
    eb = np.max(np.abs(db_a - np.asarray(db_j)))
    print(f"layer {i}:  ||ΔW||_inf = {eW:.2e}   ||Δb||_inf = {eb:.2e}")
    max_err = max(max_err, eW, eb)

print(f"\nmax |analytic − AD| over all params = {max_err:.2e}")