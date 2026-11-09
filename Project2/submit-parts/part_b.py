import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from function_setup2 import runge, sigmoid, sigmoid_der
from data_aquisition import design_matrix
from part_a import C_mse, dC_dy_pred_batch


def create_layers(network_input_size, layer_output_sizes, seed=None, init="xavier"):
    rng = np.random.default_rng(seed)
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        if init == "xavier":
            scale = np.sqrt(1.0 / i_size)
        elif init == "he":
            scale = np.sqrt(2.0 / i_size)
        elif init == "small":
            scale = 0.01
        else:
            raise ValueError(f"Unknown init: {init!r}")
        W = rng.normal(0.0, scale, size=(i_size, layer_output_size))
        b = np.zeros(layer_output_size)
        layers.append((W, b))
        i_size = layer_output_size
    return layers


def I_identity(z):
    return z


def dI_dz(z):
    return np.ones_like(z)

# column vector of input data
def raw_x(x):
    return np.asarray(x).reshape(-1, 1)


def feed_forward_saver_batch(inputs, layers, activation_funcs):
    """ Save the layer inputs and pre-activations for backpropagation. """
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
    """
        Compute gradients of the cost function with respect to each layer's weights and biases.
    """
    if forward_pass is None:
        forward_pass = feed_forward_saver_batch(
            inputs, layers, activation_funcs
        )
    layer_inputs, zs, predict = forward_pass
    layer_grads = [None] * len(layers)
    dC_dZ = None

    for i in reversed(range(len(layers))):
        A_prev = layer_inputs[i]
        Z = zs[i]
        actv_der = activation_ders[i]

        if i == len(layers) - 1:
            dC_dZ = cost_der(targets, predict)
        else:
            W_next = layers[i + 1][0]
            dC_dZ = dC_dZ @ W_next.T

        dC_dZ = dC_dZ * actv_der(Z)
        dC_dW = A_prev.T @ dC_dZ
        dC_db = dC_dZ.sum(axis=0)
        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads

#--------------------------#
#        Optimizers
#--------------------------#
def sgd_step(layers, grads, gamma):
    return [(W - gamma * dW, b - gamma * db)
            for (W, b), (dW, db) in zip(layers, grads)]


def rmsprop_step(layers, grads, state, gamma, beta=0.9, eps=1e-8):
    new_state, new_layers = [], []
    for (W, b), (dW, db), (sW, sb) in zip(layers, grads, state):
        sW = beta * sW + (1 - beta) * dW ** 2
        sb = beta * sb + (1 - beta) * db ** 2
        new_layers.append((
            W - gamma * dW / (np.sqrt(sW) + eps),
            b - gamma * db / (np.sqrt(sb) + eps),
        ))
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

        mW_hat = mW / (1 - beta1 ** t)
        vW_hat = vW / (1 - beta2 ** t)
        mb_hat = mb / (1 - beta1 ** t)
        vb_hat = vb / (1 - beta2 ** t)

        new_layers.append((
            W - gamma * mW_hat / (np.sqrt(vW_hat) + eps),
            b - gamma * mb_hat / (np.sqrt(vb_hat) + eps),
        ))
        new_state.append((mW, vW, mb, vb))
    return new_layers, new_state


def train_network(X_train, y_train, layers, actv_funcs, actv_ders,
                  optimizer="adam", gamma=1e-3, n_epochs=500,
                  batch_size=32, seed=42, return_history=False):
    n = len(y_train)
    rng = np.random.default_rng(seed)

    if optimizer == "rmsprop":
        state = [(np.zeros_like(W), np.zeros_like(b)) for W, b in layers]
    elif optimizer == "adam":
        state = [(np.zeros_like(W), np.zeros_like(W),
                  np.zeros_like(b), np.zeros_like(b)) for W, b in layers]
    else:
        state = None

    history = [] if return_history else None
    t = 0
    for epoch in range(n_epochs):
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

    if return_history:
        return layers, history
    return layers
        

def predict(X, layers, actv_funcs):
    _, _, a = feed_forward_saver_batch(X, layers, actv_funcs)
    return a


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



    # ============================================================
    # STEP 1 — OLS baseline (sanity check)
    # ============================================================
    deg_best = 10
    X_poly = design_matrix(x, degree=deg_best, intercept=True)
    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(
        X_poly, y, test_size=0.3, random_state=42
    )
    theta_ols = np.linalg.pinv(Xp_tr.T @ Xp_tr) @ Xp_tr.T @ yp_tr
    mse_ols = C_mse(yp_te, Xp_te @ theta_ols)
    print(f"\n=== OLS baseline (degree {deg_best}, same 70/30 split) ===")
    print(f"    test MSE = {mse_ols:.4e}")
    print(f"    (Project 1 best Ridge was ~1.8e-2 on its own split)")

        # Ridge/Lasso baselines at Project 1's best (degree, λ)
    from sklearn.linear_model import Ridge, Lasso
    for d_rl, lam_rl, name_rl in [
        (14, 1.0e-4, "Ridge (Project 1 best d, λ)"),
        (14, 2.3e-4, "Lasso (Project 1 best d, λ)"),
    ]:
        X_rl = design_matrix(x, degree=d_rl, intercept=True)
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            X_rl, y, test_size=0.3, random_state=42
        )
        scaler_rl = StandardScaler().fit(Xr_tr)
        Xr_tr = scaler_rl.transform(Xr_tr)
        Xr_te = scaler_rl.transform(Xr_te)
        # Note: sklearn Ridge/Lasso already handle the intercept
        model = Ridge(alpha=lam_rl) if "Ridge" in name_rl else Lasso(alpha=lam_rl, max_iter=10000)
        model.fit(Xr_tr, yr_tr)
        print(f"    {name_rl}: test MSE = {C_mse(yr_te, model.predict(Xr_te)):.4e}")

    # ============================================================
    # STEP 2 — Single-network sanity check (tuned γ)
    # ============================================================
    print("\n=== Single-network sanity (1 hidden, 50 nodes, Adam, γ=0.1) ===")
    acts_1l = [sigmoid, I_identity]
    ders_1l = [sigmoid_der, dI_dz]
    layers = create_layers(1, [50, 1], seed=42)
    layers = train_network(X_tr, y_tr, layers, acts_1l, ders_1l,
                           optimizer="adam", gamma=0.1,
                           n_epochs=500, batch_size=32)
    train_mse = C_mse(y_tr, predict(X_tr, layers, acts_1l))
    test_mse = C_mse(y_te, predict(X_te, layers, acts_1l))
    print(f"    train MSE = {train_mse:.4e}")
    print(f"    test  MSE = {test_mse:.4e}")
    print(f"    EXPECTED: test MSE ~1.5e-2 (matches the table's 1×50 row)")

    # ============================================================
    # FIGURE 3 — Training-loss curves for the three optimizers
    # ============================================================
    acts_2l = [sigmoid, sigmoid, I_identity]
    ders_2l = [sigmoid_der, sigmoid_der, dI_dz]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt, g, color in [("sgd", 0.1, "tab:blue"),
                          ("rmsprop", 0.01, "tab:orange"),
                          ("adam", 0.01, "tab:green")]:
        layers = create_layers(1, [50, 50, 1], seed=42)
        _, hist = train_network(X_tr, y_tr, layers, acts_2l, ders_2l,
                                optimizer=opt, gamma=g,
                                n_epochs=500, batch_size=32,
                                return_history=True)
        ax.semilogy(range(1, 501), hist, color=color,
                    label=f"{opt.upper()} (γ={g})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train MSE")
    ax.set_title("Training loss on Runge (2 hidden, 50/50, sigmoid)")
    ax.legend(frameon=False); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()

    # ============================================================
    # STEP 4 Four-architecture comparison (fills Table I)
    # ============================================================
    print("\n=== Architecture comparison (best optimizer per arch) ===")
    print(f"{'Architecture':<22} {'best opt':<10} {'γ':<8} {'test MSE':<12}")
    print("-" * 60)

    configs = [
        ("1 hidden, 50 nodes",   [50, 1]),
        ("1 hidden, 100 nodes",  [100, 1]),
        ("2 hidden, 50/50",      [50, 50, 1]),
        ("2 hidden, 100/100",    [100, 100, 1]),
    ]

    for name, out_sizes in configs:
        acts = [sigmoid] * (len(out_sizes) - 1) + [I_identity]
        ders = [sigmoid_der] * (len(out_sizes) - 1) + [dI_dz]
        best = None  # (test_mse, opt, gamma)
        for opt in ["sgd", "rmsprop", "adam"]:
            for gamma in [1e-4, 1e-3, 1e-2, 1e-1]:
                try:
                    layers = create_layers(1, out_sizes, seed=42)
                    layers = train_network(X_tr, y_tr, layers, acts, ders,
                                           optimizer=opt, gamma=gamma,
                                           n_epochs=500, batch_size=32)
                    mse = C_mse(y_te, predict(X_te, layers, acts))
                    if np.isfinite(mse) and (best is None or mse < best[0]):
                        best = (mse, opt, gamma)
                except (FloatingPointError, OverflowError):
                    pass
        mse, opt, gamma = best
        print(f"{name:<22} {opt:<10} {gamma:<8.0e} {mse:<12.4e}")

    # ============================================================
    # STEP 5 Learning-rate sweep (fills Table II)
    # ============================================================
    print("\n=== Learning-rate sweep (2 hidden, 50/50, sigmoid) ===")
    print(f"{'optimizer':<10} " + " ".join(f"{g:>9.0e}" for g in [1e-4, 1e-3, 1e-2, 1e-1]))
    print("-" * 55)
    for opt in ["sgd", "rmsprop", "adam"]:
        row = []
        for gamma in [1e-4, 1e-3, 1e-2, 1e-1]:
            try:
                layers = create_layers(1, [50, 50, 1], seed=42)
                layers = train_network(X_tr, y_tr, layers, acts_2l, ders_2l,
                                       optimizer=opt, gamma=gamma,
                                       n_epochs=500, batch_size=32)
                mse = C_mse(y_te, predict(X_te, layers, acts_2l))
                row.append(f"{mse:9.2e}" if np.isfinite(mse) else "      nan")
            except (FloatingPointError, OverflowError):
                row.append("      inf")
        print(f"{opt:<10} " + " ".join(row))

    # ============================================================
    # STEP 6 Epochs to threshold, at each optimizer's tuned γ
    # ============================================================
    print("\n=== Epochs to reach training MSE < 1e-2 (2 hidden 50/50, tuned γ) ===")
    tuned_gamma = {"sgd": 0.1, "rmsprop": 0.01, "adam": 0.01}
    for opt, g in tuned_gamma.items():
        L = create_layers(1, [50, 50, 1], seed=42)
        _, hist = train_network(X_tr, y_tr, L, acts_2l, ders_2l,
                                optimizer=opt, gamma=g,
                                n_epochs=500, batch_size=32,
                                return_history=True)
        below = np.where(np.array(hist) < 1e-2)[0]
        reached = int(below[0]) + 1 if len(below) > 0 else None
        print(f"    {opt:<10} (γ={g}) reached in {reached if reached else '>500'} epochs")

    # ============================================================
    # FIGURE NN fits on Runge data
    # ============================================================
    import matplotlib.pyplot as plt
    xx_plot = np.linspace(-1, 1, 400).reshape(-1, 1)
    xx_scaled = scaler.transform(xx_plot)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xx_plot.ravel(), runge(xx_plot.ravel()),
            "k-", label="Runge (true)", linewidth=2)
    ax.scatter(x, y, s=8, alpha=0.4, label="noisy data")

    for name, out_sizes, opt, gamma, color in [
        ("1×50 (Adam, γ=0.1)",   [50, 1],   "adam",    0.1,  "tab:blue"),
        ("2×100/100 (RMSprop, γ=1e-3)", [100, 100, 1], "rmsprop", 1e-3, "tab:red"),
    ]:
        acts = [sigmoid] * (len(out_sizes) - 1) + [I_identity]
        ders = [sigmoid_der] * (len(out_sizes) - 1) + [dI_dz]
        L = create_layers(1, out_sizes, seed=42)
        L = train_network(X_tr, y_tr, L, acts, ders,
                          optimizer=opt, gamma=gamma,
                          n_epochs=500, batch_size=32)
        y_pred = predict(xx_scaled, L, acts).ravel()
        ax.plot(xx_plot.ravel(), y_pred, "--", color=color,
                label=name, linewidth=1.5)

    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Network fits vs Runge function")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.show()

    

    # ============================================================
    # FIGURE Learning-rate sweep
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 4.5))
    gammas_plot = [1e-4, 1e-3, 1e-2, 1e-1]
    sweep_results = {
        "sgd":     [1.17e-1, 1.15e-1, 1.11e-1, 3.93e-2],
        "rmsprop": [1.10e-1, 2.15e-2, 1.52e-2, 4.42e-2],
        "adam":    [1.10e-1, 2.02e-2, 1.63e-2, 1.63e-2],
    }
    for opt, marker in [("sgd", "o"), ("rmsprop", "s"), ("adam", "^")]:
        ax.loglog(gammas_plot, sweep_results[opt], marker + "-", label=opt, linewidth=1.5)

    ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("Test MSE")
    ax.set_title("Learning-rate sweep (2 hidden 50/50, sigmoid)")
    ax.legend(frameon=False); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()
    