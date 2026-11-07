"""
Part c) — Activation × depth × width, with tuned γ per architecture.

Uses the tuned optimizers/learning rates from part b Table I:
  - 1 hidden layer: Adam, γ=0.1
  - 2 hidden layers: RMSprop, γ=0.01
ReLU and Leaky ReLU use He initialization; sigmoid uses Xavier.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from function_setup2 import (runge, sigmoid, sigmoid_der,
                              ReLU, ReLU_der, leaky_ReLU, leaky_ReLU_der)
from data_aquisition import design_matrix
from part_a import C_mse
from part_b import (raw_x, I_identity, dI_dz, create_layers,
                    train_network, predict)


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

    # (act, der, init) per activation
    activation_sets = {
        "Sigmoid":    (sigmoid,    sigmoid_der,    "xavier"),
        "ReLU":       (ReLU,       ReLU_der,       "he"),
        "Leaky ReLU": (leaky_ReLU, leaky_ReLU_der, "he"),
    }

    architectures = [
        ("1 hidden, 20",   [20, 1]),
        ("1 hidden, 50",   [50, 1]),
        ("1 hidden, 100",  [100, 1]),
        ("2 hidden, 20/20",    [20, 20, 1]),
        ("2 hidden, 50/50",    [50, 50, 1]),
        ("2 hidden, 100/100",  [100, 100, 1]),
    ]

    print(f"{'Activation':<12} {'Architecture':<20} "
          f"{'Optimizer':<10} {'γ':<8} {'Train MSE':<12} {'Test MSE':<12} {'# Params':<10}")
    print("-" * 90)

    results = []
    for act_name, (act_fn, act_der, init) in activation_sets.items():
        for arch_name, out_sizes in architectures:
            acts = [act_fn] * (len(out_sizes) - 1) + [I_identity]
            ders = [act_der] * (len(out_sizes) - 1) + [dI_dz]

            # Tuned optimizers, from part b Table I:
            #   shallow networks: Adam at γ=0.1
            #   deep networks:    RMSprop at γ=0.01
            if len(out_sizes) == 2:
                opt, gamma = "adam", 0.1
            else:
                opt, gamma = "rmsprop", 0.01

            try:
                layers = create_layers(1, out_sizes, seed=42, init=init)
                layers = train_network(
                    X_tr, y_tr, layers, acts, ders,
                    optimizer=opt, gamma=gamma,
                    n_epochs=500, batch_size=32, seed=42,
                )
                train_mse = C_mse(y_tr, predict(X_tr, layers, acts))
                test_mse  = C_mse(y_te, predict(X_te, layers, acts))
                n_params = sum(W.size + b.size for W, b in layers)

                # NaN / divergence guard: if either MSE is non-finite OR
                # absurdly large (say > 10), the run has diverged. Report
                # it explicitly rather than plotting 10^12 in a log graph.
                if not (np.isfinite(train_mse) and np.isfinite(test_mse)) \
                        or train_mse > 10 or test_mse > 10:
                    print(f"{act_name:<12} {arch_name:<20} {opt:<10} {gamma:<8.0e} "
                          f"DIVERGED (train={train_mse:.2e}, test={test_mse:.2e})")
                    continue

                print(f"{act_name:<12} {arch_name:<20} {opt:<10} {gamma:<8.0e} "
                      f"{train_mse:<12.4e} {test_mse:<12.4e} {n_params:<10}")

                results.append({
                    "activation": act_name,
                    "arch": arch_name,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "n_params": n_params,
                })
            except (FloatingPointError, OverflowError, ValueError) as e:
                print(f"{act_name:<12} {arch_name:<20} {opt:<10} {gamma:<8.0e} "
                      f"ERROR: {e}")

    # ---- Plot only valid results, with bounded y-axis ----
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_colors = {
        "Sigmoid":    ("darkred",   "red"),
        "ReLU":       ("darkgreen", "green"),
        "Leaky ReLU": ("darkblue",  "blue"),
    }
    for act_name in activation_sets:
        pts = [r for r in results if r["activation"] == act_name]
        if not pts:
            continue
        params = [r["n_params"] for r in pts]
        train  = [r["train_mse"] for r in pts]
        test   = [r["test_mse"] for r in pts]
        train_color, test_color = plot_colors[act_name]
        ax.plot(params, train, "o--", color=train_color, label=f"{act_name} train")
        ax.plot(params, test,  "s-",  color=test_color,  label=f"{act_name} test")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-3, 3e-1)          # <-- bound the axis to sane values
    ax.axhline(2.33e-2, color="gray", linestyle=":",
               label="OLS same-split baseline")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("MSE")
    ax.set_title("Part c) — train vs test MSE across architectures")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()