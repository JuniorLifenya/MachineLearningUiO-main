import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from Project1.Submit.function_setup import design_matrix
from fits import fit_ols_SVD, fit_ridge

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


def cv_score_degree(x,y,degree,k=5, lam = 0.0, seed = 42):
    X_all = design_matrix(x,degree,intercept=False)
    kf = KFold(n_splits=k, shuffle = True, random_state = seed)

    fold_mses = []
    for tr_idx,va_idx in kf.split(X_all):
        X_tr , X_va = X_all[tr_idx], X_all[va_idx]
        y_tr,y_va = y[tr_idx], y[va_idx]

        # For degree 0: no features so predict the mean
        if degree == 0:
            y_pred = np.full(len(y_va),y_tr.mean())
            fold_mses.append(np.mean((y_va-y_pred)**2))
            continue

        # scaler again inside fold
        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_va = scaler.transform(X_va)

        # Center y inside the fold
        y_mean = y_tr.mean()
        y_tr_c = y_tr - y_mean

        # Fit 
        if lam == 0:
            theta = fit_ols_SVD(X_tr,y_tr_c)
        else:
            theta = fit_ridge(X_tr,y_tr_c,lam)

        y_pred = X_va @ theta + y_mean
        fold_mses.append(np.mean((y_va-y_pred)**2))

    return float (np.mean(fold_mses))