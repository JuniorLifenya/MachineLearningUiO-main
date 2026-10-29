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


def train_regularized(X, y, layers, lam, penalty_grad, gamma, epochs=300):
    rng = np.random.default_rng(42)
    state = [(np.zeros_like(W), np.zeros_like(W),
              np.zeros_like(b), np.zeros_like(b)) for W, b in layers]
    acts = [sigmoid] * (len(layers) - 1) + [I_identity]
    ders = [sigmoid_der] * (len(layers) - 1) + [dI_dz]
    t = 0

    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 32):
            ids = order[start:start + 32]
            grads = backpropagation_batch(
                X[ids], layers, acts, y[ids], ders, dC_dypred_batch
            )
            reg_grads = penalty_grad([W for W, _ in layers], lam)
            grads = [(dW + reg_dW, db)
                     for (dW, db), reg_dW in zip(grads, reg_grads)]
            t += 1
            layers, state = adam_step(layers, grads, state, gamma, t)
    return layers, acts


def best_polynomial_baseline(x, y, train_ids, val_ids, test_ids, model_type):
    best = None
    for degree in range(1, 16):
        X = design_matrix(x, degree, intercept=False)
        scaler = StandardScaler().fit(X[train_ids])
        X_train = scaler.transform(X[train_ids])
        X_val = scaler.transform(X[val_ids])
        alpha_min = 1e-3 if model_type == "Lasso" else 1e-6
        for alpha in np.logspace(np.log10(alpha_min), 1, 12):
            model = (Ridge(alpha=alpha) if model_type == "Ridge"
                     else Lasso(alpha=alpha, max_iter=10000))
            model.fit(X_train, y[train_ids])
            val_mse = C_mse(y[val_ids], model.predict(X_val))
            if best is None or val_mse < best[0]:
                best = (val_mse, degree, alpha, model, scaler)
    val_mse, degree, alpha, model, scaler = best
    test_mse = C_mse(
        y[test_ids], model.predict(scaler.transform(
            design_matrix(x, degree, intercept=False)[test_ids]
        ))
    )
    return val_mse, degree, alpha, test_mse


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    x = np.sort(rng.uniform(-1.0, 1.0, 100))
    y = runge(x) + rng.normal(0.0, 0.1, len(x))
    ids_trainval, ids_test = train_test_split(
        np.arange(len(y)), test_size=0.3, random_state=42
    )
    ids_train, ids_val = train_test_split(
        ids_trainval, test_size=0.2, random_state=42
    )

    X = raw_x(x)
    scaler = StandardScaler().fit(X[ids_train])
    X_train, X_val, X_test = (
        scaler.transform(X[ids_train]),
        scaler.transform(X[ids_val]),
        scaler.transform(X[ids_test]),
    )
    y_train = y[ids_train].reshape(-1, 1)

    searches = {
        "L1": (L1_penalty_grad, [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
        "L2": (L2_penalty_grad, [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
    }
    for name, (penalty_grad, lambdas) in searches.items():
        best = None
        for architecture in ([20, 1], [20, 20, 1]):
            acts = [sigmoid] * (len(architecture) - 1) + [I_identity]
            for gamma in (1e-3, 3e-3):
                for lam in lambdas:
                    layers, acts = train_regularized(
                        X_train, y_train,
                        create_layers(1, architecture, seed=42),
                        lam, penalty_grad, gamma,
                    )
                    val_mse = C_mse(
                        y[ids_val].reshape(-1, 1), predict(X_val, layers, acts)
                    )
                    if best is None or val_mse < best[0]:
                        best = (val_mse, architecture, gamma, lam, layers, acts)
        val_mse, architecture, gamma, lam, layers, acts = best
        test_mse = C_mse(
            y[ids_test].reshape(-1, 1), predict(X_test, layers, acts)
        )
        print(f"NN {name}: layers={architecture}, gamma={gamma:g}, "
              f"lambda={lam:g}, validation MSE={val_mse:.4e}, "
              f"test MSE={test_mse:.4e}")

    for model_type in ("Ridge", "Lasso"):
        val_mse, degree, alpha, test_mse = best_polynomial_baseline(
            x, y, ids_train, ids_val, ids_test, model_type
        )
        print(f"Project 1 {model_type}: degree={degree}, alpha={alpha:.2e}, "
              f"validation MSE={val_mse:.4e}, test MSE={test_mse:.4e}")