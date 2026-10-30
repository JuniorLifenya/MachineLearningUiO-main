import os

from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
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


def train_mnist(X_train, y_train, X_val, y_val, hidden_sizes,
                activation_funcs, activation_ders,
                gamma=1e-3, epochs=10, batch_size=128, seed=42):
    """Train a softmax classifier using the provided network architecture."""
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

            gradients = backpropagation_batch(
                X_batch,
                layers,
                activation_funcs,
                y_batch,
                activation_ders,
                _cross_entropy_d_logits,
                forward_pass=forward_pass,
            )
            step += 1
            layers, state = adam_step(layers, gradients, state, gamma, step)

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

    activation_funcs = [leaky_ReLU, leaky_ReLU, softmax]
    activation_ders = [leaky_ReLU_der, leaky_ReLU_der, dI_dz]
    layers, activation_funcs, _ = train_mnist(
        X_train, y_train, X_val, y_val,
        hidden_sizes=(256, 128),
        activation_funcs=activation_funcs,
        activation_ders=activation_ders,
        gamma=2e-3,
        epochs=50,
        batch_size=256,
        seed=42
    )

    y_test_one_hot = np.eye(10, dtype=np.float32)[y_test]
    test_loss, test_accuracy = _evaluate(
        X_test, y_test_one_hot, layers, activation_funcs
    )
    print(f"Test loss={test_loss:.4f}, test accuracy={test_accuracy:.4f}")

    """CLASSIFICATION REPORT"""

    from sklearn.metrics import (confusion_matrix as sk_confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score,
                             classification_report as sk_classification_report,
                             roc_curve as sk_roc_curve, roc_auc_score, RocCurveDisplay)

    # from week 43 exercises - classification report and confusion matrix
    threshold = 0.5
    p_mnist = np.asarray(feed_forward_saver_batch(X_test, layers, activation_funcs)[2])
    y_pred_mnist = (p_mnist > threshold).astype(int)
    y_pred_mnist = np.argmax(p_mnist, axis=1) # for multi-class classification, we take the class with the highest probability
    y_test = np.asarray(y_test).ravel()

    """ROC CURVE AND AUC"""

    """CONFUSION MATRIX DISPLAY"""
    cm = sk_confusion_matrix(y_test, y_pred_mnist, labels=np.arange(10))
    print(cm)

    digit_totals = cm.sum(axis=1)
    digit_accuracies = np.divide(
        np.diag(cm),
        digit_totals,
        out=np.zeros(len(digit_totals), dtype=float),
        where=digit_totals != 0,
    )
    print("Accuracy per digit:")
    for digit, accuracy in enumerate(digit_accuracies):
        print(f"Digit {digit}: {accuracy:.2%}")
    
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=np.arange(10),
    )
    display.plot(cmap="Greens", values_format="d")
    plt.show()


    
