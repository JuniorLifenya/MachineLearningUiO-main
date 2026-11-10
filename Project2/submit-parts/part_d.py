import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from function_setup2 import (
    runge, sigmoid, sigmoid_der, ReLU, ReLU_der, leaky_ReLU, leaky_ReLU_der,
)
from data_aquisition import design_matrix
from part_a import C_mse, dC_dy_pred_batch, L1_penalty_grad, L2_penalty_grad
from part_b import (
    I_identity, dI_dz, raw_x, create_layers, backpropagation_batch,
    adam_step, predict,
)


def train_regularized(
    X_train, y_train, layers, lam, penalty_grad, gamma,
    epochs=300, batch_size=32, seed=42, verbose=True, model_name="NN",
    hidden_activation=sigmoid, hidden_activation_derivative=sigmoid_der,
):
    """Train a regularized regression network and record its threshold epoch."""
    rng = np.random.default_rng(seed)
    state = [
        (np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b))
        for W, b in layers
    ]
    activation_funcs = [hidden_activation] * (len(layers) - 1) + [I_identity]
    activation_ders = (
        [hidden_activation_derivative] * (len(layers) - 1) + [dI_dz]
    )
    t = 0
    threshold_epoch = None

    for epoch in range(epochs):
        order = rng.permutation(len(y_train))
        for start in range(0, len(y_train), batch_size):
            indices = order[start:start + batch_size]
            grads = backpropagation_batch(
                X_train[indices], layers, activation_funcs, y_train[indices],
                activation_ders, dC_dy_pred_batch,
            )
            reg_grads = penalty_grad([W for W, _ in layers], lam)
            grads = [
                (dW + reg_dW, db)
                for (dW, db), reg_dW in zip(grads, reg_grads)
            ]
            t += 1
            layers, state = adam_step(layers, grads, state, gamma, t)
        train_mse = C_mse(y_train, predict(X_train, layers, activation_funcs))
        reached_threshold = train_mse < 1e-2
        if reached_threshold and threshold_epoch is None:
            threshold_epoch = epoch + 1
        if verbose:
            threshold_status = (
                " (reached MSE < 1e-2)" if reached_threshold
                and threshold_epoch == epoch + 1 else ""
            )
            print(
                f"{model_name}: optimizer=Adam, epoch={epoch + 1}/{epochs}, "
                f"train MSE={train_mse:.4e}, gamma={gamma:.3e}, "
                f"lambda={lam:.3e}, batch_size={batch_size}, seed={seed}"
                f"{threshold_status}"
            )
    return layers, activation_funcs, threshold_epoch


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


def architecture_name(layer_sizes):
    hidden_sizes = layer_sizes[:-1]
    if len(set(hidden_sizes)) == 1:
        return f"{len(hidden_sizes)}x{hidden_sizes[0]}"
    return "x".join(map(str, hidden_sizes))


if __name__ == "__main__":
    seed = 42
    data_rng = np.random.default_rng(2026)
    x = np.sort(data_rng.uniform(-1.0, 1.0, 100))
    y = runge(x) + data_rng.normal(0.0, 0.1, len(x))
    X = raw_x(x)

    # Split the data into training, validation, and test sets
    trainval_indices, test_indices = train_test_split(
        np.arange(len(y)), test_size=0.3, random_state=seed
    )
    #training data split for hyperparameters and weights in seperate sets
    train_indices, val_indices = train_test_split(
        trainval_indices, test_size=0.2, random_state=seed
    )

    scaler = StandardScaler().fit(X[train_indices])
    X_train, X_val, X_test = (
        scaler.transform(X[train_indices]),
        scaler.transform(X[val_indices]),
        scaler.transform(X[test_indices]),
    )
    y_train = y[train_indices].reshape(-1, 1)
    y_val = y[val_indices].reshape(-1, 1)
    y_test = y[test_indices].reshape(-1, 1)

    epochs = 100
    batch_size = 64
    lambdas = np.concatenate(
        ([0.0], np.logspace(np.log10(1e-8), np.log10(1e-0), 11))
    )
    gammas = np.logspace(np.log10(1e-2), np.log10(1e0), 12)

    architectures = (
        (20, 20, 1),
        (100, 100, 1),
        (200, 200, 20, 1),
    )
    nn_architectures = {
        architecture_name(architecture): architecture
        for architecture in architectures
    }
    searches = {
        "L1": (L1_penalty_grad, lambdas),
        "L2": (L2_penalty_grad, lambdas),
    }

    mse_grids = {}
    arch_best = {}
    for name, (penalty_grad, search_lambdas) in searches.items():
        mse_grid = np.full((len(gammas), len(search_lambdas)), np.inf)
        arch_best[name] = {}

        for arch_name, architecture in nn_architectures.items():
            for gamma_index, gamma in enumerate(gammas):
                for lambda_index, lam in enumerate(search_lambdas):
                    layers, activation_funcs, threshold_epoch = train_regularized(
                        X_train,
                        y_train,
                        create_layers(1, architecture, seed=seed),
                        lam=lam,
                        penalty_grad=penalty_grad,
                        gamma=gamma,
                        epochs=epochs,
                        batch_size=batch_size,
                        seed=seed,
                        verbose=True,
                        model_name=f"NN {name} layers={architecture}",
                    )
                    val_mse = C_mse(
                        y_val, predict(X_val, layers, activation_funcs)
                    )
                    # The heatmap records the best value across architectures.
                    mse_grid[gamma_index, lambda_index] = min(
                        mse_grid[gamma_index, lambda_index], val_mse
                    )
                    # per-architecture best for the scatter
                    if (arch_name not in arch_best[name]
                            or val_mse < arch_best[name][arch_name][2]):
                        arch_best[name][arch_name] = (
                            gamma_index, lambda_index, val_mse, gamma, lam,
                            layers, activation_funcs, threshold_epoch,
                        )
        mse_grids[name] = mse_grid
        best_arch_name, best_config = min(
            arch_best[name].items(), key=lambda item: item[1][2]
        )
        (
            gamma_index, lambda_index, val_mse, gamma, lam, layers,
            activation_funcs, _,
        ) = best_config
        architecture = nn_architectures[best_arch_name]
        test_mse = C_mse(
            y_test, predict(X_test, layers, activation_funcs)
        )
        print(f"NN {name}: layers={architecture}, gamma={gamma:g}, "
              f"lambda={lam:g}, validation MSE={val_mse:.4e}, "
              f"test MSE={test_mse:.4e}")

    for model_type in ("Ridge", "Lasso"):
        val_mse, degree, alpha, test_mse = best_polynomial_baseline(
            x, y, train_indices, val_indices, test_indices, model_type
        )
        print(f"Project 1 {model_type}: degree={degree}, alpha={alpha:.2e}, "
              f"validation MSE={val_mse:.4e}, test MSE={test_mse:.4e}")

    # Creating heatmap plot for hyperparameters

    fig, axes = plt.subplots(
        1, len(searches), figsize=(14, 7), sharey=True
    )
    vmin = min(grid.min() for grid in mse_grids.values())
    vmax = max(grid.max() for grid in mse_grids.values())

    marker_shapes = ("s", "D", "^", "o", "v", "P", "X")
    marker_colors = (
        "red", "crimson", "black", "darkorange", "blue", "purple", "brown"
    )
    markers = {
        name: (
            marker_shapes[index % len(marker_shapes)],
            marker_colors[index % len(marker_colors)],
        )
        for index, name in enumerate(nn_architectures)
    }

    for ax, (name, (_, search_lambdas)) in zip(axes, searches.items()):
        image = ax.imshow(
            mse_grids[name], origin="lower", aspect="auto", cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"{name} regularization (best over architectures)")
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(np.arange(len(search_lambdas)))
        ax.set_xticklabels(
            [f"{lam:.0e}" for lam in search_lambdas], rotation=45, fontsize=8
        )
        ax.set_yticks(np.arange(len(gammas)))
        ax.set_yticklabels([f"{gamma:.0e}" for gamma in gammas], fontsize=8)

        # Scatter marks: one per architecture, showing WHICH cell that arch prefers.
        for arch_name, best_config in arch_best[name].items():
            gi, li = best_config[:2]
            mk, col = markers[arch_name]
            ax.scatter(li, gi, marker=mk, s=180, facecolors="none",
                       edgecolors=col, linewidths=2.2,
                       label=f"{arch_name} best")

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            fontsize=9,
            framealpha=0.9,
        )

    axes[0].set_ylabel(r"Learning rate $\gamma$")
    fig.suptitle(r"Validation MSE over $(\gamma, \lambda)$ with per-architecture optimum marked")
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.23, top=0.88, wspace=0.3)
    colorbar_ax = fig.add_axes([0.89, 0.28, 0.02, 0.54])
    fig.colorbar(
        image, cax=colorbar_ax, label="Validation MSE (min over architectures)"
    )
    plt.show()

    # Print for the report table
    for name, dict_by_arch in arch_best.items():
        print(f"\n{name}: best result for each architecture")
        for arch_name, config in dict_by_arch.items():
            (
                _, _, val_mse, gamma, lam, layers, activation_funcs,
                threshold_epoch,
            ) = config
            test_mse = C_mse(
                y_test, predict(X_test, layers, activation_funcs)
            )
            epoch_result = (
                f"{threshold_epoch}/{epochs}"
                if threshold_epoch is not None else f"not reached in {epochs}"
            )
            print(
                f"  {arch_name} (layers={nn_architectures[arch_name]}): "
                f"optimizer=Adam, epoch to train MSE < 1e-2={epoch_result}, "
                f"gamma={gamma:.3e}, lambda={lam:.3e}, "
                f"batch_size={batch_size}, seed={seed}, "
                f"validation MSE={val_mse:.4e}, test MSE={test_mse:.4e}"
            )

    activation_sets = {
        "Sigmoid": (sigmoid, sigmoid_der, "xavier"),
        "ReLU": (ReLU, ReLU_der, "he"),
        "Leaky ReLU": (leaky_ReLU, leaky_ReLU_der, "he"),
    }
    print("\n=== Epoch to reach training MSE < 1e-2 by activation ===")
    print("ReLU and Leaky ReLU use each architecture's best sigmoid gamma/lambda.")
    for name, dict_by_arch in arch_best.items():
        print(f"\n{name}:")
        for arch_name, config in dict_by_arch.items():
            (
                _, _, _, gamma, lam, _, _, sigmoid_threshold_epoch,
            ) = config
            for activation_name, (
                activation, activation_derivative, initialization,
            ) in activation_sets.items():
                if activation_name == "Sigmoid":
                    threshold_epoch = sigmoid_threshold_epoch
                else:
                    _, _, threshold_epoch = train_regularized(
                        X_train,
                        y_train,
                        create_layers(
                            1,
                            nn_architectures[arch_name],
                            seed=seed,
                            init=initialization,
                        ),
                        lam=lam,
                        penalty_grad=searches[name][0],
                        gamma=gamma,
                        epochs=epochs,
                        batch_size=batch_size,
                        seed=seed,
                        verbose=True,
                        model_name=(
                            f"NN {name} {activation_name} "
                            f"layers={nn_architectures[arch_name]}"
                        ),
                        hidden_activation=activation,
                        hidden_activation_derivative=activation_derivative,
                    )
                epoch_result = (
                    str(threshold_epoch)
                    if threshold_epoch is not None
                    else f"not reached in {epochs}"
                )
                print(
                    f"  {activation_name:<12} {arch_name:<12} "
                    f"epoch={epoch_result}, gamma={gamma:.3e}, "
                    f"lambda={lam:.3e}, batch_size={batch_size}, seed={seed}"
                )