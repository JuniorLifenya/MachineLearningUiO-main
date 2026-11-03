import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Allow this script to be run from any working directory.
module_dir = str(Path(__file__).resolve().parent)
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from function_setup2 import leaky_ReLU, leaky_ReLU_der
from part_a import (
    L1_penalty_grad,
    L2_penalty_grad,
    dC_dz_softmax,
    multiclass_cross_entropy,
    softmax,
)
from part_b import (
    backpropagation_batch,
    create_layers,
    dI_dz,
    adam_step,
    predict,
    feed_forward_saver_batch,
)


def load_mnist():
    """Load MNIST with pixel values in [0, 1] and 784 features per image."""
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidates = [
        os.path.join(here, "mnist.npz"),
        os.path.join(here, "..", "BookPrograms", "chapter10_convolutional_networks", "mnist.npz"),
        os.path.join(here, "..", "..", "BookML", "BookPrograms", "chapter10_convolutional_networks", "mnist.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            d = np.load(path)
            X_train = d["x_train"].reshape(len(d["x_train"]), -1).astype(float) / 255.0
            X_test = d["x_test"].reshape(len(d["x_test"]), -1).astype(float) / 255.0
            return X_train, d["y_train"].astype(int), X_test, d["y_test"].astype(int)

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(float) / 255.0
    y = mnist.target.astype(int)
    return X[:60000], y[:60000], X[60000:], y[60000:]


def _evaluate(X, y, layers, activation_funcs):
    probabilities = np.asarray(predict(X, layers, activation_funcs))
    loss = float(multiclass_cross_entropy(y, probabilities))
    accuracy = float(np.mean(np.argmax(probabilities, axis=1) == np.argmax(y, axis=1)))
    return loss, accuracy


def _classification_metrics(y_true, probabilities):
    labels = np.arange(10)
    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(probabilities)
    y_pred = np.argmax(probabilities, axis=1)
    y_true_one_hot = np.eye(len(labels), dtype=int)[y_true]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    digit_totals = cm.sum(axis=1)
    digit_accuracies = np.divide(
        np.diag(cm),
        digit_totals,
        out=np.zeros(len(digit_totals), dtype=float),
        where=digit_totals != 0,
    )
    return {
        "predictions": y_pred,
        "confusion_matrix": cm,
        "digit_accuracies": digit_accuracies,
        "auc_per_digit": roc_auc_score(y_true_one_hot, probabilities, average=None),
        "auc_macro": roc_auc_score(y_true_one_hot, probabilities, average="macro"),
    }


def _add_model_config_captions(fig, axes, results, epochs, batch_size):
    for ax, (_, metrics) in zip(axes, results.items()):
        config = metrics["configuration"]
        position = ax.get_position()
        fig.text(
            position.x0 + position.width / 2,
            0.025,
            f"Hidden layers: {config['hidden_layers']}\n"
            f"Activations: {config['activations']}\n"
            f"Regularization: {config['regularization']} "
            f"(lambda={config['lam']:.1e}) | Learning rate: {config['gamma']:.1e}\n"
            f"Epochs: {epochs} | Batch size: {batch_size}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def train_mnist(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_sizes,
    activation_funcs,
    activation_ders,
    lam=0.0,
    penalty_grad=None,
    gamma=1e-3,
    epochs=10,
    batch_size=128,
    seed=42,
    verbose=True,
):
    """Train a softmax classifier using Adam and optional L1 or L2 regularization."""
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be a finite, non-negative value")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite, positive learning rate")
    if lam > 0 and penalty_grad is None:
        raise ValueError("provide a regularization gradient when lam is positive")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    input_size = X_train.shape[1]
    hidden_sizes = tuple(hidden_sizes)
    activation_funcs = list(activation_funcs)
    activation_ders = list(activation_ders)
    if X_val.shape[1] != input_size:
        raise ValueError("training and validation data must have the same feature count")

    num_classes = 10
    layer_sizes = (*hidden_sizes, num_classes)
    if len(activation_funcs) != len(layer_sizes):
        raise ValueError("provide one activation function for each network layer")
    if len(activation_ders) != len(layer_sizes):
        raise ValueError("provide one activation derivative for each network layer")

    y_train_one_hot = np.eye(num_classes, dtype=np.float32)[y_train]
    y_val_one_hot = np.eye(num_classes, dtype=np.float32)[y_val]
    layers = create_layers(input_size, layer_sizes, seed=seed)

    # Adam tracks moving averages of gradients and squared gradients.
    state = [
        (np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b))
        for W, b in layers
    ]
    rng = np.random.default_rng(seed)
    step = 0
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        order = rng.permutation(len(y_train_one_hot))
        loss_sum = 0.0
        correct = 0

        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            X_batch = X_train[indices]
            y_batch = y_train_one_hot[indices]
            forward_pass = feed_forward_saver_batch(X_batch, layers, activation_funcs)
            probabilities = np.asarray(forward_pass[2])
            loss_sum += float(multiclass_cross_entropy(y_batch, probabilities)) * len(indices)
            correct += int(np.sum(np.argmax(probabilities, axis=1) == np.argmax(y_batch, axis=1)))

            grads = backpropagation_batch(
                X_batch,
                layers,
                activation_funcs,
                y_batch,
                activation_ders,
                dC_dz_softmax,
                forward_pass=forward_pass,
            )

            # Add the regularization derivative to weight gradients, not bias gradients.
            if lam > 0:
                reg_grads = penalty_grad([W for W, _ in layers], lam)
                if len(reg_grads) != len(grads) or any(
                    reg_dW.shape != dW.shape
                    for (dW, _), reg_dW in zip(grads, reg_grads)
                ):
                    raise ValueError("regularization gradients must match each weight matrix")
                grads = [
                    (dW + reg_dW, db)
                    for (dW, db), reg_dW in zip(grads, reg_grads)
                ]

            step += 1
            layers, state = adam_step(layers, grads, state, gamma, step)

        history["train_loss"].append(loss_sum / len(order))
        history["train_accuracy"].append(correct / len(order))
        val_loss, val_accuracy = _evaluate(X_val, y_val_one_hot, layers, activation_funcs)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train loss={history['train_loss'][-1]:.4f}, "
                f"train accuracy={history['train_accuracy'][-1]:.4f}, "
                f"validation loss={val_loss:.4f}, "
                f"validation accuracy={val_accuracy:.4f}"
            )

    return layers, activation_funcs, history


def plot_hyperparameter_heatmap(lambdas, gammas, accuracy_grids, best_indices):
    """Compare validation accuracy grids for L1 and L2 regularization."""
    min_accuracy = min(np.min(grid) for grid in accuracy_grids.values())
    max_accuracy = max(np.max(grid) for grid in accuracy_grids.values())
    if min_accuracy == max_accuracy:
        max_accuracy = min_accuracy + 1e-12

    fig, axes = plt.subplots(
        1, len(accuracy_grids), figsize=(16, 6), sharey=True,
        constrained_layout=True,
    )
    images = []
    for ax, (name, accuracies) in zip(axes, accuracy_grids.items()):
        image = ax.imshow(
            accuracies,
            cmap="viridis",
            aspect="auto",
            vmin=min_accuracy,
            vmax=max_accuracy,
        )
        images.append(image)
        ax.set_xticks(
            np.arange(len(lambdas)), [f"{lam:.0e}" for lam in lambdas]
        )
        ax.set_yticks(
            np.arange(len(gammas)), [f"{gamma:.0e}" for gamma in gammas]
        )
        ax.set_xlabel("Lambda (lambda)")
        ax.set_title(f"{name} regularization")

        for i in range(len(gammas)):
            for j in range(len(lambdas)):
                score = accuracies[i, j]
                ax.text(
                    j, i, f"{score:.3f}", ha="center", va="center",
                    color="white" if score < 0.55 else "black", fontsize=9,
                )

        best_i, best_j = best_indices[name]
        ax.scatter(
            best_j, best_i, marker="s", s=1350, facecolors="none",
            edgecolors="red", linewidths=2,
        )
    axes[0].set_ylabel("Gamma (gamma)")
    fig.colorbar(images[0], ax=axes, label="Validation accuracy")
    fig.suptitle("Validation accuracy by gamma and lambda")
    return fig


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    epochs = 30
    batch_size = 2024
    seed = 42
    logistic_gamma = 2e-3
    lambdas = np.concatenate(([0.0], np.logspace(np.log10(1e-12), np.log10(5e-6), 5)))
    gammas = np.logspace(np.log10(1e-3), np.log10(5e-2), 5)

    activation_funcs_nn = [leaky_ReLU, leaky_ReLU, softmax]
    activation_ders_nn = [leaky_ReLU_der, leaky_ReLU_der, dI_dz]
    activation_funcs_log = [softmax]
    activation_ders_log = [dI_dz]

    searches = {
        "L1": (L1_penalty_grad, lambdas),
        "L2": (L2_penalty_grad, lambdas),
    }
    validation_accuracies = {
        name: np.zeros((len(gammas), len(search_lambdas)))
        for name, (_, search_lambdas) in searches.items()
    }
    best_indices = {}
    best_models = {}

    # Select each penalty's hyperparameters by validation accuracy only.
    for name, (penalty_grad, search_lambdas) in searches.items():
        best_model = None
        for lambda_index, lam in enumerate(search_lambdas):
            for gamma_index, gamma in enumerate(gammas):
                print(
                    f"Training neural network ({name}): "
                    f"lambda={lam:.1e}, gamma={gamma:.1e}"
                )
                layers, activation_funcs, history = train_mnist(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    hidden_sizes=(128, 64),
                    activation_funcs=activation_funcs_nn,
                    activation_ders=activation_ders_nn,
                    lam=float(lam),
                    penalty_grad=penalty_grad,
                    gamma=float(gamma),
                    epochs=epochs,
                    batch_size=batch_size,
                    seed=seed,
                    verbose=True,
                )
                val_accuracy = history["val_accuracy"][-1]
                validation_accuracies[name][gamma_index, lambda_index] = (
                    val_accuracy
                )
                print(f"  Final validation accuracy: {val_accuracy:.4f}")

                if best_model is None or val_accuracy > best_model[0]:
                    best_model = (
                        val_accuracy,
                        (gamma_index, lambda_index),
                        float(lam),
                        float(gamma),
                        layers,
                        activation_funcs,
                    )

        best_models[name] = best_model
        best_indices[name] = best_model[1]
        print(
            f"Best {name} model: lambda={best_model[2]:.1e}, "
            f"gamma={best_model[3]:.1e}, "
            f"validation accuracy={best_model[0]:.4f}"
        )

    best_regularization = max(
        best_models, key=lambda name: best_models[name][0]
    )
    (
        best_accuracy,
        _,
        best_lam,
        best_gamma,
        best_layers_nn,
        best_activation_funcs_nn,
    ) = best_models[best_regularization]
    print(
        f"\nBest neural network: {best_regularization}, "
        f"lambda={best_lam:.1e}, "
        f"gamma={best_gamma:.1e}, "
        f"validation accuracy={best_accuracy:.4f}"
    )

    plot_hyperparameter_heatmap(
        lambdas, gammas, validation_accuracies, best_indices
    )

    print("\nTraining logistic regression (no hidden layers)")
    layers_log, funcs_log, _ = train_mnist(
        X_train,
        y_train,
        X_val,
        y_val,
        hidden_sizes=(),
        activation_funcs=activation_funcs_log,
        activation_ders=activation_ders_log,
        gamma=logistic_gamma,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    y_test_one_hot = np.eye(10, dtype=np.float32)[y_test]
    models = {
        "Logistic regression": (layers_log, funcs_log),
        "Best neural network": (best_layers_nn, best_activation_funcs_nn),
    }
    model_configurations = {
        "Logistic regression": {
            "hidden_layers": "None",
            "activations": "Softmax",
            "regularization": "None",
            "lam": 0.0,
            "gamma": logistic_gamma,
        },
        "Best neural network": {
            "hidden_layers": "128, 64",
            "activations": "Leaky ReLU, Leaky ReLU, Softmax",
            "regularization": best_regularization,
            "lam": best_lam,
            "gamma": best_gamma,
        },
    }
    results = {}
    for model_name, (layers, activation_funcs) in models.items():
        probabilities = np.asarray(predict(X_test, layers, activation_funcs))
        test_loss = float(multiclass_cross_entropy(y_test_one_hot, probabilities))
        metrics = _classification_metrics(y_test, probabilities)
        results[model_name] = {
            "probabilities": probabilities,
            "configuration": model_configurations[model_name],
            **metrics,
        }

        print(f"\n{model_name} test loss: {test_loss:.4f}")
        print(f"{model_name} test accuracy: {np.mean(metrics['predictions'] == y_test):.4f}")
        print(f"{model_name} macro ROC AUC: {metrics['auc_macro']:.4f}")
        print("ROC AUC per digit:")
        for digit, auc in enumerate(metrics["auc_per_digit"]):
            print(f"  Digit {digit}: {auc:.4f}")
        print("Classification report:")
        print(
            classification_report(
                y_test,
                metrics["predictions"],
                labels=np.arange(10),
                target_names=[str(digit) for digit in range(10)],
                zero_division=0,
            )
        )
        print("Confusion matrix:")
        print(metrics["confusion_matrix"])
        print("Accuracy per digit:")
        for digit, accuracy in enumerate(metrics["digit_accuracies"]):
            print(f"  Digit {digit}: {accuracy:.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (model_name, metrics) in zip(axes, results.items()):
        display = ConfusionMatrixDisplay(
            confusion_matrix=metrics["confusion_matrix"],
            display_labels=np.arange(10),
        )
        cmap = "Blues" if model_name == "Logistic regression" else "Greens"
        display.plot(ax=ax, cmap=cmap, values_format="d", colorbar=False)
        ax.set_title(model_name)
    fig.suptitle("MNIST confusion matrices")
    fig.tight_layout(rect=(0, 0.24, 1, 0.94))
    _add_model_config_captions(fig, axes, results, epochs, batch_size)

    y_test_one_hot = np.eye(10, dtype=int)[np.asarray(y_test).ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    roc_colors = plt.get_cmap("viridis", 10)
    for ax, (model_name, metrics) in zip(axes, results.items()):
        for digit in range(10):
            fpr, tpr, _ = roc_curve(
                y_test_one_hot[:, digit],
                metrics["probabilities"][:, digit],
            )
            ax.plot(
                fpr,
                tpr,
                color=roc_colors(digit),
                label=f"Digit {digit} (AUC {metrics['auc_per_digit'][digit]:.3f})",
            )
        ax.plot([0, 1], [0, 1], "k--", label="Chance")
        ax.set(
            title=f"{model_name} (macro AUC {metrics['auc_macro']:.3f})",
            xlabel="False positive rate",
            ylabel="True positive rate",
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("MNIST one-vs-rest ROC curves")
    fig.tight_layout(rect=(0, 0.24, 1, 0.94))
    _add_model_config_captions(fig, axes, results, epochs, batch_size)
    plt.show()
