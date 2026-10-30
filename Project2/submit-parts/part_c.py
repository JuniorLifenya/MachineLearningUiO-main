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
from part_a import C_mse, dC_dy_pred_batch
from part_b import raw_x,I_identity,dI_dz,create_layers,train_network,predict

# ================================================================================
# PART 3 — Driver for part c): activation × depth × width
# 1.Sweep hidden actv from sigmoid to ReLu, then Leaky ReLu
# 2.Sweep depth and width: 1vs2 hidden layers,20 vs 50 vs 100 nodes
# 3.Look for overfitting: plot training vs test Mse as a func of total params
# 4.Discuss winning actv function, why, where we see bias-var tradeoff in the #
# ================================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 100
    sigma = 0.1
    x = np.sort(rng.uniform(-1.0, 1.0, n))          # FIXED
    y = runge(x) + rng.normal(0.0, sigma, n)

    # NN input
    X_nn = raw_x(x)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_nn, y.reshape(-1, 1), test_size=0.3, random_state=42
    )
    scaler = StandardScaler().fit(X_tr)              # FIXED
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    # --- OLS baseline ---
    deg_best = 10
    X_poly = design_matrix(x, degree=deg_best, intercept=True)
    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(
        X_poly, y, test_size=0.3, random_state=42
    )
    theta_ols = np.linalg.pinv(Xp_tr.T @ Xp_tr) @ Xp_tr.T @ yp_tr
    mse_ols = C_mse(yp_te, Xp_te @ theta_ols)
    print(f"OLS (degree {deg_best}): test MSE = {mse_ols:.4e}\n")

    # --- Activation pairs ---
    activation_sets = {
        "Sigmoid":    (sigmoid,    sigmoid_der),
        "ReLU":       (ReLU,       ReLU_der),
        "Leaky ReLU": (leaky_ReLU, leaky_ReLU_der),
    }

    # --- Architecture sweep ---
    # Each entry: (name, layer_output_sizes)
    architectures = [
        ("1 hidden, 20",   [20, 1]),
        ("1 hidden, 50",   [50, 1]),
        ("1 hidden, 100",  [100, 1]),
        ("2 hidden, 20/20",   [20, 20, 1]),
        ("2 hidden, 50/50",   [50, 50, 1]),
        ("2 hidden, 100/100", [100, 100, 1]),
    ]

    # --- Header ---
    print(f"{'Activation':<12} {'Architecture':<20} "
          f"{'Train MSE':<12} {'Test MSE':<12} {'# Params':<10}")
    print("-" * 70)

    results = []
    for act_name, (act_fn, act_der) in activation_sets.items():
        for arch_name, out_sizes in architectures:
            # Build activations: all hidden use act_fn, output is linear
            acts = [act_fn] * (len(out_sizes) - 1) + [I_identity]
            ders = [act_der] * (len(out_sizes) - 1) + [dI_dz]

            layers = create_layers(1, out_sizes, seed=42)
            layers = train_network(
                X_tr, y_tr, layers, acts, ders,
                optimizer="adam", gamma=1e-3,
                n_epochs=1000, batch_size=32, seed=42,
            )

            train_mse = C_mse(y_tr, predict(X_tr, layers, acts))
            test_mse  = C_mse(y_te, predict(X_te, layers, acts))
            n_params = sum(W.size + b.size for W, b in layers)

            print(f"{act_name:<12} {arch_name:<20} "
                  f"{train_mse:<12.4e} {test_mse:<12.4e} {n_params:<10}")

            results.append({
                "activation": act_name,
                "arch": arch_name,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "n_params": n_params,
            })

    # --- Optional: plot train vs test MSE vs parameters ---
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_colors = {
        "Sigmoid": ("darkred", "red"),
        "ReLU": ("darkgreen", "green"),
        "Leaky ReLU": ("darkblue", "blue"),
    }
    for act_name in activation_sets:
        pts = [r for r in results if r["activation"] == act_name]
        params = [r["n_params"] for r in pts]
        train  = [r["train_mse"] for r in pts]
        test   = [r["test_mse"] for r in pts]
        train_color, test_color = plot_colors[act_name]
        ax.plot(params, train, "o--", color=train_color, label=f"{act_name} train")
        ax.plot(params, test,  "o-",  color=test_color, label=f"{act_name} test")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("MSE")
    ax.set_title("Part c) — train vs test MSE across architectures")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()