import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from Part_ab import generate_data, runge, design_matrix
from function_setup import fit_ols_SVD,fit_ridge


# ==================================================================
# Training and Test MSE. Part C) Note
# For each polynomial degree, fit OLS many times on bootstrap 
# samples of the training data,then compare the variability and mean
# of predictions against fixed test set
# ==================================================================

def bootstrap_bias_variance(x, y, degree, B=100, test_size=0.3, seed=42):
    # --- Split once ---
    X_all = design_matrix(x, degree, intercept=False)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y, test_size=test_size, random_state=seed
    )

    # --- Scale once, fit on train, apply to test ---
    # We do NOT refit the scaler inside the bootstrap loop. The bootstrap
    # is simulating resampled training data, not resampled preprocessing.
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    # --- Center y once (the model has no intercept column) ---
    y_mean = y_tr.mean()
    y_tr_c = y_tr - y_mean

    # --- Bootstrap loop ---
    rng = np.random.default_rng(seed)
    n_tr = len(y_tr)
    preds = np.zeros((B, len(y_te)))
    for b in range(B):
        idx = rng.integers(0, n_tr, size=n_tr)   # with replacement
        theta = fit_ols_SVD(X_tr[idx], y_tr_c[idx])
        preds[b] = X_te @ theta + y_mean

    # --- Decompose ---
    mean_pred = preds.mean(axis=0)
    bias_sq = np.mean((y_te - mean_pred) ** 2)
    variance = np.mean(preds.var(axis=0))

    return {
        "degree": degree,
        "bias_sq": bias_sq,
        "variance": variance,
        "mse": bias_sq + variance,
        "preds": preds,
        "y_test": y_te,
    }

def plot_bias_variance(results):
    d = [r["degree"] for r in results]
    b = [r["bias_sq"] for r in results]
    v = [r["variance"] for r in results]
    m = [r["mse"] for r in results]

    plt.plot(d, b, "o-", label=r"Bias$^2$ (+σ²)")
    plt.plot(d, v, "s-", label="Variance")
    plt.plot(d, m, "^-", label="Total")
    plt.yscale("log")
    plt.xlabel("Degree"); plt.legend(); plt.grid(True)
    plt.show()

x,y = generate_data()
results = [
    bootstrap_bias_variance(x,y,degree=d)
    for d in range(1,6)
]

plot_bias_variance(results)