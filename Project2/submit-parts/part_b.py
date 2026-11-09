import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge                       # Lasso removed (§3)
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from function_setup2 import runge, sigmoid, sigmoid_der
from data_aquisition import design_matrix
from part_a import C_mse, dC_dy_pred_batch


# -------------------- network code (unchanged) --------------------
def create_layers(network_input_size, layer_output_sizes, seed=None, init="xavier"):
    rng = np.random.default_rng(seed)
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        scale = {"xavier": np.sqrt(1.0 / i_size),
                 "he":     np.sqrt(2.0 / i_size),
                 "small":  0.01}.get(init)
        if scale is None:
            raise ValueError(f"Unknown init: {init!r}")
        W = rng.normal(0.0, scale, size=(i_size, layer_output_size))
        b = np.zeros(layer_output_size)
        layers.append((W, b))
        i_size = layer_output_size
    return layers


def I_identity(z):          return z
def dI_dz(z):               return np.ones_like(z)
def raw_x(x):               return np.asarray(x).reshape(-1, 1)


def feed_forward_saver_batch(inputs, layers, activation_funcs):
    layer_inputs, zs = [], []
    a = inputs
    for (W, b), activation in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W + b
        a = activation(z)
        zs.append(z)
    return layer_inputs, zs, a


def backpropagation_batch(inputs, layers, activation_funcs, targets,
                          activation_ders, cost_der, forward_pass=None):
    if forward_pass is None:
        forward_pass = feed_forward_saver_batch(inputs, layers, activation_funcs)
    layer_inputs, zs, predict_ = forward_pass
    layer_grads = [None] * len(layers)
    dC_dZ = None
    for i in reversed(range(len(layers))):
        A_prev, Z, actv_der = layer_inputs[i], zs[i], activation_ders[i]
        if i == len(layers) - 1:
            # Fragile-by-design: correct ONLY because final activation is I_identity.
            # If you swap the output activation, multiply by actv_der(Z) here.
            dC_dZ = cost_der(targets, predict_)
        else:
            dC_dZ = dC_dZ @ layers[i + 1][0].T
        dC_dZ = dC_dZ * actv_der(Z)
        layer_grads[i] = (A_prev.T @ dC_dZ, dC_dZ.sum(axis=0))
    return layer_grads


# -------------------- optimizers (unchanged) --------------------
def sgd_step(layers, grads, gamma):
    return [(W - gamma * dW, b - gamma * db)
            for (W, b), (dW, db) in zip(layers, grads)]


def rmsprop_step(layers, grads, state, gamma, beta=0.9, eps=1e-8):
    new_state, new_layers = [], []
    for (W, b), (dW, db), (sW, sb) in zip(layers, grads, state):
        sW = beta * sW + (1 - beta) * dW ** 2
        sb = beta * sb + (1 - beta) * db ** 2
        new_layers.append((W - gamma * dW / (np.sqrt(sW) + eps),
                           b - gamma * db / (np.sqrt(sb) + eps)))
        new_state.append((sW, sb))
    return new_layers, new_state


def adam_step(layers, grads, state, gamma, t,
              beta1=0.9, beta2=0.999, eps=1e-8):
    new_state, new_layers = [], []
    for (W, b), (dW, db), (mW, vW, mb, vb) in zip(layers, grads, state):
        mW = beta1 * mW + (1 - beta1) * dW
        vW = beta2 * vW + (1 - beta2) * dW ** 2
        mb = beta1 * mb + (1 - beta1) * db
        vb = beta2 * vb + (1 - beta2) * db ** 2
        mW_hat, vW_hat = mW / (1 - beta1 ** t), vW / (1 - beta2 ** t)
        mb_hat, vb_hat = mb / (1 - beta1 ** t), vb / (1 - beta2 ** t)
        new_layers.append((W - gamma * mW_hat / (np.sqrt(vW_hat) + eps),
                           b - gamma * mb_hat / (np.sqrt(vb_hat) + eps)))
        new_state.append((mW, vW, mb, vb))
    return new_layers, new_state


# -------------------- training (backward compatible, adds X_val) --------------------
def train_network(X_train, y_train, layers, actv_funcs, actv_ders,
                  optimizer="adam", gamma=1e-3, n_epochs=500,
                  batch_size=32, seed=42, return_history=False,
                  X_val=None, y_val=None):
    n = len(y_train)
    rng = np.random.default_rng(seed)

    if optimizer == "rmsprop":
        state = [(np.zeros_like(W), np.zeros_like(b)) for W, b in layers]
    elif optimizer == "adam":
        state = [(np.zeros_like(W), np.zeros_like(W),
                  np.zeros_like(b), np.zeros_like(b)) for W, b in layers]
    else:
        state = None

    history     = [] if return_history else None
    val_history = [] if (return_history and X_val is not None) else None
    t = 0
    for _ in range(n_epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            Xb = X_train[idx[start:start + batch_size]]
            yb = y_train[idx[start:start + batch_size]]
            grads = backpropagation_batch(
                Xb, layers, actv_funcs, yb, actv_ders, dC_dy_pred_batch
            )
            if optimizer == "sgd":
                layers = sgd_step(layers, grads, gamma)
            elif optimizer == "rmsprop":
                layers, state = rmsprop_step(layers, grads, state, gamma)
            elif optimizer == "adam":
                t += 1
                layers, state = adam_step(layers, grads, state, gamma, t)
        if return_history:
            history.append(C_mse(y_train, predict(X_train, layers, actv_funcs)))
            if val_history is not None:
                val_history.append(C_mse(y_val, predict(X_val, layers, actv_funcs)))

    if return_history:
        if val_history is not None:
            return layers, history, val_history
        return layers, history
    return layers


def predict(X, layers, actv_funcs):
    _, _, a = feed_forward_saver_batch(X, layers, actv_funcs)
    return a


# -------------------- NEW: CV utilities (§1) --------------------
def kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return [
        (np.concatenate([folds[j] for j in range(k) if j != i]), folds[i])
        for i in range(k)
    ]


def cv_score(X_raw, y, out_sizes, acts, ders, optimizer, gamma,
             n_epochs=500, batch_size=32, k=5, seed=42):
    fold_mses = []
    for tr_idx, val_idx in kfold_indices(len(y), k, seed):
        sc = StandardScaler().fit(X_raw[tr_idx])
        L = create_layers(X_raw.shape[1], out_sizes, seed=seed)
        L = train_network(sc.transform(X_raw[tr_idx]), y[tr_idx],
                          L, acts, ders,
                          optimizer=optimizer, gamma=gamma,
                          n_epochs=n_epochs, batch_size=batch_size)
        fold_mses.append(
            C_mse(y[val_idx], predict(sc.transform(X_raw[val_idx]), L, acts))
        )
    return float(np.mean(fold_mses)), float(np.std(fold_mses))


def select_hyperparams_cv(X_raw, y, out_sizes, acts, ders,
                          optimizers=("sgd", "rmsprop", "adam"),
                          gammas=(1e-4, 1e-3, 1e-2, 1e-1),
                          n_epochs=500, batch_size=32, k=5, seed=42):
    results = []
    for opt in optimizers:
        for g in gammas:
            try:
                m, s = cv_score(X_raw, y, out_sizes, acts, ders,
                                optimizer=opt, gamma=g,
                                n_epochs=n_epochs, batch_size=batch_size,
                                k=k, seed=seed)
            except (FloatingPointError, OverflowError):
                m, s = np.inf, np.nan
            results.append((m, s, opt, g))
    results.sort(key=lambda r: r[0])
    return results


def fit_final(X_raw, y, out_sizes, acts, ders, opt, gamma,
              n_epochs=500, batch_size=32, seed=42):
    sc = StandardScaler().fit(X_raw)
    L = create_layers(X_raw.shape[1], out_sizes, seed=seed)
    L = train_network(sc.transform(X_raw), y, L, acts, ders,
                      optimizer=opt, gamma=gamma,
                      n_epochs=n_epochs, batch_size=batch_size)
    return L, sc


# ============================================================
if __name__ == "__main__":

    rng   = np.random.default_rng(2026)
    n     = 100
    sigma = 0.1
    x     = rng.uniform(-1.0, 1.0, n)          # §8a: no sort
    y     = runge(x) + rng.normal(0.0, sigma, n)

    # Canonical raw split — never overwrite; CV and refits own their scalers (§7)
    X_nn = raw_x(x)
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_nn, y.reshape(-1, 1), test_size=0.3, random_state=42
    )

    # ============================================================
    # STEP 1 — OLS / Ridge baselines (no test leakage, no dead intercept column)
    # ============================================================
    print("\n=== Polynomial baselines (same 70/30 split) ===")
    deg_best = 10
    X_poly = design_matrix(x, degree=deg_best, intercept=True)   # OLS fits b explicitly
    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(
        X_poly, y, test_size=0.3, random_state=42
    )
    theta_ols = np.linalg.pinv(Xp_tr.T @ Xp_tr) @ Xp_tr.T @ yp_tr
    print(f"    OLS (deg {deg_best}): test MSE = "
          f"{C_mse(yp_te, Xp_te @ theta_ols):.4e}")

    d_rl, lam_rl = 14, 1.0e-4
    X_rl = design_matrix(x, degree=d_rl, intercept=False)   # §4: sklearn adds b
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_rl, y, test_size=0.3, random_state=42
    )
    sc_rl = StandardScaler().fit(Xr_tr)
    ridge = Ridge(alpha=lam_rl).fit(sc_rl.transform(Xr_tr), yr_tr)
    print(f"    Ridge (d={d_rl}, α={lam_rl:.1e}): test MSE = "
          f"{C_mse(yr_te, ridge.predict(sc_rl.transform(Xr_te))):.4e}")
    # Lasso removed (§3): fails to converge at this α on degree-14 features.

    # ============================================================
    # STEP 2 — Single-network sanity check
    # ============================================================
    print("\n=== Single-network sanity (1 hidden ×50, Adam γ=0.1) ===")
    acts_1l = [sigmoid, I_identity]
    ders_1l = [sigmoid_der, dI_dz]
    L, sc = fit_final(X_tr_raw, y_tr, [50, 1], acts_1l, ders_1l,
                      opt="adam", gamma=0.1)
    print(f"    train MSE = {C_mse(y_tr, predict(sc.transform(X_tr_raw), L, acts_1l)):.4e}")
    print(f"    test  MSE = {C_mse(y_te, predict(sc.transform(X_te_raw), L, acts_1l)):.4e}")

    # ============================================================
    # FIGURE 3 — Optimizer comparison (loss curves only; no test-set use)
    # ============================================================
    acts_2l = [sigmoid, sigmoid, I_identity]
    ders_2l = [sigmoid_der, sigmoid_der, dI_dz]
    sc_2l = StandardScaler().fit(X_tr_raw)
    X_tr_2l = sc_2l.transform(X_tr_raw)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt, g, color in [("sgd", 0.1, "tab:blue"),
                          ("rmsprop", 0.01, "tab:orange"),
                          ("adam", 0.01, "tab:green")]:
        L = create_layers(1, [50, 50, 1], seed=42)
        _, hist = train_network(X_tr_2l, y_tr, L, acts_2l, ders_2l,
                                optimizer=opt, gamma=g,
                                n_epochs=500, batch_size=32,
                                return_history=True)
        ax.semilogy(range(1, 501), hist, color=color,
                    label=f"{opt.upper()} (γ={g})")
    ax.set(xlabel="Epoch", ylabel="Train MSE",
           title="Training loss on Runge (2 hidden, 50/50, sigmoid)")
    ax.legend(frameon=False); ax.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.show()

    # ============================================================
    # STEP 4 — Architecture comparison via 5-fold CV
    # ============================================================
    print("\n=== Architecture comparison (CV-selected hyperparameters) ===")
    print(f"{'Architecture':<22} {'opt':<9} {'γ':<7} "
          f"{'CV MSE (mean±std)':<22} {'Test MSE':<10}")
    print("-" * 78)

    configs = [
        ("1 hidden, 50 nodes",   [50, 1]),
        ("1 hidden, 100 nodes",  [100, 1]),
        ("2 hidden, 50/50",      [50, 50, 1]),
        ("2 hidden, 100/100",    [100, 100, 1]),
    ]

    for name, out_sizes in configs:
        acts = [sigmoid] * (len(out_sizes) - 1) + [I_identity]
        ders = [sigmoid_der] * (len(out_sizes) - 1) + [dI_dz]

        ranked = select_hyperparams_cv(X_tr_raw, y_tr, out_sizes, acts, ders)
        cv_m, cv_s, opt, gamma = ranked[0]

        L, sc = fit_final(X_tr_raw, y_tr, out_sizes, acts, ders, opt, gamma)
        test_mse = C_mse(y_te, predict(sc.transform(X_te_raw), L, acts))

        print(f"{name:<22} {opt:<9} {gamma:<7.0e} "
              f"{cv_m:.3e} ± {cv_s:.1e}      {test_mse:.3e}")

    # ============================================================
    # STEP 5 — LR sweep (5-fold CV, reused for the figure)
    # ============================================================
    print("\n=== LR sweep via 5-fold CV (2 hidden 50/50, sigmoid) ===")
    sweep_gammas = [1e-4, 1e-3, 1e-2, 1e-1]
    sweep_results, sweep_errors = {}, {}

    print(f"{'optimizer':<10} " + " ".join(f"{g:>13.0e}" for g in sweep_gammas))
    print("-" * 70)
    for opt in ("sgd", "rmsprop", "adam"):
        means, stds = [], []
        for g in sweep_gammas:
            try:
                m, s = cv_score(X_tr_raw, y_tr, [50, 50, 1],
                                acts_2l, ders_2l,
                                optimizer=opt, gamma=g, k=5, seed=42)
            except (FloatingPointError, OverflowError):
                m, s = np.inf, np.nan
            means.append(m); stds.append(s)
        sweep_results[opt] = means
        sweep_errors[opt]  = stds
        print(f"{opt:<10} " + " ".join(
            f"{m:>7.2e}±{s:<4.1e}" for m, s in zip(means, stds)
        ))

    # ============================================================
    # STEP 6 — Epochs to reach a 1.5× noise-floor target
    # ============================================================
    noise_floor = sigma ** 2
    threshold   = 1.5 * noise_floor
    print(f"\n=== Epochs to reach TRAIN MSE < {threshold:.1e} "
          f"(1.5× σ² = {noise_floor:.1e}) — 2 hidden 50/50, tuned γ ===")
    tuned_gamma = {"sgd": 0.1, "rmsprop": 0.01, "adam": 0.01}
    for opt, g in tuned_gamma.items():
        L = create_layers(1, [50, 50, 1], seed=42)
        _, hist = train_network(X_tr_2l, y_tr, L, acts_2l, ders_2l,
                                optimizer=opt, gamma=g,
                                n_epochs=500, batch_size=32,
                                return_history=True)
        below = np.where(np.array(hist) < threshold)[0]
        reached = int(below[0]) + 1 if len(below) else None
        print(f"    {opt:<10} (γ={g}) reached in "
              f"{reached if reached is not None else '>500'} epochs")

    # ============================================================
    # FIGURE — Fits vs Runge (fit_final ensures the plot matches the table)
    # ============================================================
    xx_plot = np.linspace(-1, 1, 400).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xx_plot.ravel(), runge(xx_plot.ravel()),
            "k-", linewidth=2, label="Runge (true)")
    ax.scatter(x, y, s=8, alpha=0.4, label="noisy data")

    for name, out_sizes, opt, gamma, color in [
        ("1×50 Adam γ=0.1",              [50, 1],       "adam",    0.1,  "tab:blue"),
        ("2×100/100 RMSprop γ=1e-3",     [100, 100, 1], "rmsprop", 1e-3, "tab:red"),
    ]:
        acts = [sigmoid] * (len(out_sizes) - 1) + [I_identity]
        ders = [sigmoid_der] * (len(out_sizes) - 1) + [dI_dz]
        L, sc = fit_final(X_tr_raw, y_tr, out_sizes, acts, ders, opt, gamma)
        ax.plot(xx_plot.ravel(),
                predict(sc.transform(xx_plot), L, acts).ravel(),
                "--", color=color, linewidth=1.5, label=name)
    ax.set(xlabel="x", ylabel="y", title="Network fits vs Runge function")
    ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()

    # ============================================================
    # FIGURE — LR sweep with fold-wise error bars (same array as the table)
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt, marker in [("sgd", "o"), ("rmsprop", "s"), ("adam", "^")]:
        ax.errorbar(sweep_gammas, sweep_results[opt],
                    yerr=sweep_errors[opt],
                    marker=marker, linewidth=1.5, capsize=3, label=opt)
    ax.set(xscale="log", yscale="log",
           xlabel=r"$\gamma$", ylabel="CV MSE (mean ± std over 5 folds)",
           title="Learning-rate sweep (2 hidden 50/50, sigmoid)")
    ax.legend(frameon=False); ax.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.show()