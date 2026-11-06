import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from function_setup2 import runge, sigmoid, sigmoid_der
from data_aquisition import design_matrix
from part_a import C_mse, dC_dy_pred_batch, L1_penalty_grad, L2_penalty_grad
from part_b import (
    I_identity, dI_dz, raw_x, create_layers, backpropagation_batch,
    adam_step, predict,
)


def train_regularized(
    X, y, layers, lam, penalty_grad, gamma, epochs=300, batch_size=32, seed=42
):
    """Train a sigmoid network for regression with L1 or L2 weight regularization."""
    rng = np.random.default_rng(seed)
    state = [
        (np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b))
        for W, b in layers
    ]
    acts = [sigmoid] * (len(layers) - 1) + [I_identity]
    ders = [sigmoid_der] * (len(layers) - 1) + [dI_dz]
    t = 0

    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            ids = order[start:start + batch_size]
            grads = backpropagation_batch(
                X[ids], layers, acts, y[ids], ders, dC_dy_pred_batch
            )
            reg_grads = penalty_grad([W for W, _ in layers], lam)
            grads = [
                (dW + reg_dW, db)
                for (dW, db), reg_dW in zip(grads, reg_grads)
            ]
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
    # initialize random number generator for reproducibility
    rng = np.random.default_rng(2026)
    x = np.sort(rng.uniform(-1.0, 1.0, 100))
    y = runge(x) + rng.normal(0.0, 0.1, len(x))

    # Split the data into training, validation, and test sets
    ids_trainval, ids_test = train_test_split(
        np.arange(len(y)), test_size=0.3, random_state=42
    )
    #training data split for hyperparameters and weights in seperate sets
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

    #searches are for regularization hyperparameters, gammas are for learning rates

    #logspace functions are used to create a range of values for hyperparameter tunning
    #like linspace, but with a logarithmic scale, which is useful for exploring a wide range of values

    searches = {
        "L1": (L1_penalty_grad, np.logspace(np.log10(1e-10), np.log10(1e-2), 8)),
        "L2": (L2_penalty_grad, np.logspace(np.log10(1e-10), np.log10(1e-2), 8)),
    }
    gammas = np.logspace(np.log10(1e-2), np.log10(1e0), 8)
    epochs = 300
    batch_size = 32
    seed = 42

    mse_grids = {}
    arch_best = {} # {name: (gamma_idx, lambda_idx, val_mse, gamma, lam, arch_str)}
    for name, (penalty_grad, lambdas) in searches.items():
        mse_grid = np.full((len(gammas), len(lambdas)), np.inf)
        arch_best[name] = None

        for arch_name, architecture in [("1x20",[20,1]), ("2x20",[20,20,1]),("3x20",[20,20,20,1])]:
            acts = [sigmoid] * (len(architecture) - 1) + [I_identity]
            for gamma_index, gamma in enumerate(gammas):
                for lambda_index, lam in enumerate(lambdas):
                    layers, acts = train_regularized(
                        X_train,
                        y_train,
                        create_layers(1, architecture, seed=seed),
                        lam=lam,
                        penalty_grad=penalty_grad,
                        gamma=gamma,
                        epochs=epochs,
                        batch_size=batch_size,
                        seed=seed,
                    )
                    val_mse = C_mse(
                        y[ids_val].reshape(-1, 1), predict(X_val, layers, acts)
                    )
                    # min-heatmap still records the best of the two architectures
                    mse_grid[gamma_index, lambda_index] = min(
                        mse_grid[gamma_index, lambda_index], val_mse
                    )
                    # per-architecture best for the scatter
                    if (arch_name not in arch_best[name]
                            or val_mse < arch_best[name][arch_name][2]):
                        arch_best[name][arch_name] = (
                            gamma_index, lambda_index, val_mse, gamma, lam
                        )
        mse_grids[name] = mse_grid
        val_mse, architecture, gamma, lam, layers, acts = arch_best[name]
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

    # Creating heatmap plot for hyperparameters

    fig, axes = plt.subplots(
        1, len(searches), figsize=(13, 6), sharey=True, constrained_layout=True
    )
    vmin = min(grid.min() for grid in mse_grids.values())
    vmax = max(grid.max() for grid in mse_grids.values())

    markers = {"1×20": ("s", "red"), "2×20/20": ("D", "white")}

    for ax, (name, (_, lambdas)) in zip(axes, searches.items()):
        image = ax.imshow(
            mse_grids[name], origin="lower", aspect="auto", cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"{name} regularization (best over architectures)")
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(np.arange(len(lambdas)))
        ax.set_xticklabels([f"{lam:.0e}" for lam in lambdas], rotation=45, fontsize=8)
        ax.set_yticks(np.arange(len(gammas)))
        ax.set_yticklabels([f"{gamma:.0e}" for gamma in gammas], fontsize=8)

        # Scatter marks: one per architecture, showing WHICH cell that arch prefers.
        for arch_name, (gi, li, vm, gm, lm) in arch_best[name].items():
            mk, col = markers[arch_name]
            ax.scatter(li, gi, marker=mk, s=180, facecolors="none",
                       edgecolors=col, linewidths=2.2,
                       label=f"{arch_name} best (γ={gm:.0e}, λ={lm:.0e}, val={vm:.3e})")

        ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

    axes[0].set_ylabel(r"Learning rate $\gamma$")
    fig.colorbar(image, ax=axes, label="Validation MSE (min over architectures)")
    fig.suptitle(r"Validation MSE over $(\gamma, \lambda)$ with per-architecture optimum marked")
    plt.show()

    # Print for the report table
    for name, dict_by_arch in arch_best.items():
        print(f"\n{name}: per-architecture optimum")
        for arch_name, (gi, li, vm, gm, lm) in dict_by_arch.items():
            print(f"  {arch_name}: γ={gm:.3e}, λ={lm:.3e}, val MSE={vm:.4e}")
