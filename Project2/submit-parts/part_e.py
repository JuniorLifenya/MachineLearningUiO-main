import os

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

from function_setup2 import ReLU, ReLU_der, leaky_ReLU, leaky_ReLU_der
from part_a import multiclass_cross_entropy, softmax
from part_b import (
    backpropagation_batch,
    create_layers,
    dI_dz, 
    adam_step,
    predict,
    feed_forward_saver_batch
)


def load_mnist():
    """The 70 000 MNIST images as (X_train, y_train, X_test, y_test), pixels scaled to [0, 1], 784 columns.

    Looks for the repository's mnist.npz first (no download needed) and falls
    back on scikit-learn's fetch_openml, which downloads 70 000 images once.
    """
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidates = [os.path.join(here, "mnist.npz"),
                  os.path.join(here, "..", "BookPrograms", "chapter10_convolutional_networks", "mnist.npz"),
                  os.path.join(here, "..", "..", "BookML", "BookPrograms", "chapter10_convolutional_networks", "mnist.npz")]
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


def _cross_entropy_d_logits(y_true, y_pred):
    # Fused softmax/cross-entropy derivative; the final activation derivative
    # is set to one so backpropagation_batch applies this directly to logits.
    return (y_pred - y_true) / len(y_true)


def _evaluate(X, y, layers, activation_funcs):
    probabilities = predict(X, layers, activation_funcs)
    loss = float(multiclass_cross_entropy(y, probabilities))
    accuracy = float(
        np.mean(np.argmax(np.asarray(probabilities), axis=1) == np.argmax(y, axis=1))
    )
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
        "auc_per_digit": roc_auc_score(
            y_true_one_hot, probabilities, average=None
        ),
        "auc_macro": roc_auc_score(
            y_true_one_hot, probabilities, average="macro"
        ),
    }


def _add_model_config_captions(fig, axes, results, epochs, batch_size):
    for ax, (model_name, metrics) in zip(axes, results.items()):
        config = metrics["configuration"]
        position = ax.get_position()
        fig.text(
            position.x0 + position.width / 2,
            0.025,
            f"Hidden layers: {config['hidden_layers']}\n"
            f"Activations: {config['activations']}\n"
            f"Epochs: {epochs} | Batch size: {batch_size}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def train_mnist(X_train, y_train, X_val, y_val, hidden_sizes,
                activation_funcs, activation_ders, lam=0.0,
                regularization_ders=None,
                gamma=1e-3, epochs=10, batch_size=128, seed=42):
    """Train a softmax classifier, optionally regularizing its weights."""
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be a finite, non-negative value")
    if lam > 0 and regularization_ders is None:
        raise ValueError("provide a regularization gradient when lam is positive")

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

    #identity matrix for one-hot classification identities
    """kind of picks out the right row for the one-hot encoding of the labels
    and then uses that to index into the identity matrix to get the one-hot encoded labels"""
    y_train_one_hot = np.eye(num_classes, dtype=np.float32)[y_train]
    y_val_one_hot = np.eye(num_classes, dtype=np.float32)[y_val]

    layers = create_layers(input_size, layer_sizes, seed=seed)

    """State is a list of tuples, where each tuple contains four arrays: 
    m_W, v_W, m_b, and v_b used to store the moving averages of the 
    gradients and squared gradients for the weights and biases of each 
    layer in the neural network. The moving averages are used in the 
    Adam optimization algorithm to update the weights/biases during training."""

    state = [
        (np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b))
        for W, b in layers
    ]

    rng = np.random.default_rng(seed)
    step = 0
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        order = rng.permutation(len(y_train_one_hot)) # randomly shuffle the training data for each epoch
        loss_sum = 0.0
        correct = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            X_batch = X_train[indices]
            y_batch = y_train_one_hot[indices]
            forward_pass = feed_forward_saver_batch(
                X_batch, layers, activation_funcs
            )
            probabilities = forward_pass[2]
            loss_sum += float(multiclass_cross_entropy(y_batch, probabilities)) * len(indices)
            correct += int(
                np.sum(
                    np.argmax(np.asarray(probabilities), axis=1)
                    == np.argmax(y_batch, axis=1)
                )
            )

            grads = backpropagation_batch(
                X_batch,
                layers,
                activation_funcs,
                y_batch,
                activation_ders,
                _cross_entropy_d_logits,
                forward_pass=forward_pass,
            )
            
            # regularization
            """if reagularization constant > 0, we compute the gradient for the regularization
            and sum it with the backprop gradient to get a final gradient with regularization"""
            
            if lam > 0:
                reg_grads = regularization_ders(
                    [W for W, _ in layers], lam
                )
                if len(reg_grads) != len(grads) or any(
                    reg_dW.shape != dW.shape
                    for (dW, _), reg_dW in zip(grads, reg_grads)
                ):
                    raise ValueError(
                        "regularization gradients must match each weight matrix"
                    )
                grads = [
                    (dW + reg_dW, db)
                    for (dW, db), reg_dW in zip(grads, reg_grads)
                ]
            step += 1
            layers, state = adam_step(layers, grads, state, gamma, step)

        history["train_loss"].append(loss_sum / len(order))
        history["train_accuracy"].append(correct / len(order))
        val_loss, val_accuracy = _evaluate(
            X_val, y_val_one_hot, layers, activation_funcs
        )
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train loss={history['train_loss'][-1]:.4f}, "
            f"train accuracy={history['train_accuracy'][-1]:.4f}, "
            f"validation loss={val_loss:.4f}, "
            f"validation accuracy={val_accuracy:.4f}"
        )

    return layers, activation_funcs, history


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    epochs = 50
    batch_size = 256
    gamma = 2e-3
    seed = 42

    activation_funcs_nn = [leaky_ReLU, leaky_ReLU, softmax]
    activation_ders_nn = [leaky_ReLU_der, leaky_ReLU_der, dI_dz]

    activation_funcs_log = [softmax]
    activation_ders_log = [dI_dz]
    
    print("Training neural network (2 hidden layers: 256, 128)")
    layers_nn, activation_funcs_nn, _ = train_mnist(
        X_train, y_train, X_val, y_val,
        hidden_sizes=(256, 128),
        activation_funcs=activation_funcs_nn,
        activation_ders=activation_ders_nn,
        gamma=gamma,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed
    )

    print("Training logistic regression (softmax layer, no hidden layers)")
    layers_log, activation_funcs_log, _ = train_mnist(
        X_train, y_train, X_val, y_val,
        hidden_sizes=(),
        activation_funcs=activation_funcs_log,
        activation_ders=activation_ders_log,
        gamma=gamma,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed
    )

    y_test_one_hot = np.eye(10, dtype=np.float32)[y_test]
    models = {
        "Logistic regression": (layers_log, activation_funcs_log),
        "Neural network": (layers_nn, activation_funcs_nn),
    }
    model_configurations = {
        "Logistic regression": {
            "hidden_layers": "None",
            "activations": "Softmax",
        },
        "Neural network": {
            "hidden_layers": "256, 128",
            "activations": "Leaky ReLU, Leaky ReLU, Softmax",
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
        print(f"{model_name} test accuracy: {metrics['predictions'].eq if False else np.mean(metrics['predictions'] == y_test):.4f}")
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (model_name, metrics) in zip(axes, results.items()):
        display = ConfusionMatrixDisplay(
            confusion_matrix=metrics["confusion_matrix"],
            display_labels=np.arange(10),
        )
        cmap = "Blues" if model_name == "Logistic regression" else "Greens"
        display.plot(ax=ax, cmap=cmap, values_format="d", colorbar=False)
        ax.set_title(model_name)
    fig.suptitle("MNIST confusion matrices")
    fig.tight_layout(rect=(0, 0.2, 1, 0.94))
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
    fig.tight_layout(rect=(0, 0.2, 1, 0.94))
    _add_model_config_captions(fig, axes, results, epochs, batch_size)
    plt.show()


    
