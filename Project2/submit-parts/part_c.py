"""
Part c) — Activation × depth × width, with CV-tuned (opt, γ) per cell.

Each cell (activation, architecture) is tuned INDEPENDENTLY by 5-fold CV.
This is the only fair way to compare activations: a learning rate that is
good for sigmoid (γ=0.1 with Adam) is a dying-neuron regime for ReLU.

Reports: CV mean ± std (primary, low-variance), single-split train/test
(secondary), parameter count.

ReLU and Leaky ReLU use He initialization; sigmoid uses Xavier. This is
threaded through `init=` in `select_hyperparams_cv` and `fit_final`.
"""

import numpy as np
import matplotlib.pyplot as plt
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
                    train_network, predict,
                    select_hyperparams_cv, fit_final)      # ← new imports


# ---------------------------------------------------------------------------
# Search space: SGD was never competitive in part_b, so drop it for speed.
# 2 optimizers × 3 gammas × 5 folds × 6 architectures × 3 activations
#   = 540 trainings of 500 epochs on 70 points. ~10–20 min on a laptop.
# If too slow: reduce k to 3, or n_epochs to 300.
# ---------------------------------------------------------------------------
OPTIMIZERS = ("adam", "rmsprop")
GAMMAS     = (1e-3, 1e-2, 1e-1)


if __name__ == "__main__":
    rng   = np.random.default_rng(2026)
    n     = 100
    sigma = 0.1
    x     = rng.uniform(-1.0, 1.0, n)          # matches part_b — no sort
    y     = runge(x) + rng.normal(0.0, sigma, n)

    X_nn = raw_x(x)
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_nn, y.reshape(-1, 1), test_size=0.3, random_state=42
    )
    # Raw arrays kept raw — scaling is handled by select_hyperparams_cv
    # (fit_scaler inside each fold) and fit_final (fit_scaler on all of train).

    activation_sets = {
        "Sigmoid":    (sigmoid,    sigmoid_der,    "xavier"),
        "ReLU":       (ReLU,       ReLU_der,       "he"),
        "Leaky ReLU": (leaky_ReLU, leaky_ReLU_der, "he"),
    }
    architectures = [
        ("1×20",     [20, 1]),
        ("1×50",     [50, 1]),
        ("1×100",    [100, 1]),
        ("2×20/20",  [20, 20, 1]),
        ("2×50/50",  [50, 50, 1]),
        ("2×100/100",[100, 100, 1]),
    ]

    # ---- OLS baseline, computed from the SAME split (not a stale 2.33e-2) --
    X_poly = design_matrix(x, degree=10, intercept=True)
    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(
        X_poly, y, test_size=0.3, random_state=42
    )
    theta_ols = np.linalg.pinv(Xp_tr.T @ Xp_tr) @ Xp_tr.T @ yp_tr
    ols_mse = float(C_mse(yp_te, Xp_te @ theta_ols))

    # ---- Table header ------------------------------------------------------
    print(f"\nOLS baseline on this split: {ols_mse:.4e}\n")
    print(f"{'Activation':<12} {'Arch':<10} {'opt':<9} {'γ':<7} "
          f"{'CV MSE (mean±std)':<20} {'Train':<10} {'Test':<10} {'#Par':<6}")
    print("-" * 90)

    results = []
    for act_name, (act_fn, act_der, init) in activation_sets.items():
        for arch_name, out_sizes in architectures:
            acts = [act_fn] * (len(out_sizes) - 1) + [I_identity]
            ders = [act_der] * (len(out_sizes) - 1) + [dI_dz]

            # CV-select (opt, γ) for THIS activation × architecture
            ranked = select_hyperparams_cv(
                X_tr_raw, y_tr, out_sizes, acts, ders,
                optimizers=OPTIMIZERS, gammas=GAMMAS,
                k=5, seed=42, init=init,
            )
            cv_m, cv_s, opt, gamma = ranked[0]

            # Refit on all of X_tr_raw at the selected γ
            L, sc = fit_final(X_tr_raw, y_tr, out_sizes, acts, ders,
                              opt, gamma, seed=42, init=init)
            X_tr_sc = sc.transform(X_tr_raw)
            X_te_sc = sc.transform(X_te_raw)
            train_mse = float(C_mse(y_tr, predict(X_tr_sc, L, acts)))
            test_mse  = float(C_mse(y_te, predict(X_te_sc, L, acts)))
            n_params  = sum(W.size + b.size for W, b in L)

            # Sanity: mark diverged cells instead of printing 10^12
            if not (np.isfinite(cv_m) and np.isfinite(test_mse)) \
                    or cv_m > 1.0 or test_mse > 1.0:
                print(f"{act_name:<12} {arch_name:<10} {opt:<9} {gamma:<7.0e} "
                      f"DIVERGED (cv={cv_m:.1e}, test={test_mse:.1e})")
                continue

            print(f"{act_name:<12} {arch_name:<10} {opt:<9} {gamma:<7.0e} "
                  f"{cv_m:.3e}±{cv_s:.1e}       "
                  f"{train_mse:.3e}   {test_mse:.3e}   {n_params}")
            results.append({
                "activation": act_name, "arch": arch_name,
                "cv_m": cv_m, "cv_s": cv_s,
                "train_mse": train_mse, "test_mse": test_mse,
                "n_params": n_params, "opt": opt, "gamma": gamma,
            })

    # ---- Plot: CV mean ± std as primary; single-split test as markers ------
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"Sigmoid": "tab:red", "ReLU": "tab:green", "Leaky ReLU": "tab:blue"}
    markers = {"Sigmoid": "o", "ReLU": "s", "Leaky ReLU": "^"}

    for act_name in activation_sets:
        pts = [r for r in results if r["activation"] == act_name]
        if not pts:
            continue
        params = [r["n_params"] for r in pts]
        cv_m   = np.array([r["cv_m"]  for r in pts])
        cv_s   = np.array([r["cv_s"]  for r in pts])
        test   = np.array([r["test_mse"] for r in pts])

        # CV mean ± std — primary curve
        ax.errorbar(params, cv_m, yerr=cv_s,
                    color=colors[act_name], marker=markers[act_name],
                    capsize=3, linewidth=1.4,
                    label=f"{act_name} (CV ± std)")
        # Single-split test MSE — thin, dashed, same colour, same marker
        ax.plot(params, test, ":", color=colors[act_name], alpha=0.6,
                label=f"{act_name} (single-split test)")

    ax.axhline(ols_mse, color="gray", linestyle="--", linewidth=1.2,
               label=f"OLS same-split ({ols_mse:.2e})")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(5e-3, 5e-1)
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("MSE")
    ax.set_title("Part c) — activation × architecture, CV-tuned (opt, γ) per cell")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()