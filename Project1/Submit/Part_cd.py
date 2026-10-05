import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from Part_ab import generate_data, runge, design_matrix
from function_setup import fit_ols_SVD,fit_ridge


# ==================================================================
#           Part C) Notes Training and Test MSE. 
#     For each polynomial degree, fit OLS many times on bootstrap 
# samples of the training data,then compare the variability and mean
#           of predictions against fixed test set
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


# ==================================================================
#           Part d) Notes. CV and one Fold    
#       Train on the indices NOT in the fold
#           Predict on the indices IN the fold
#                   MSE on the fold 
# ==================================================================

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

# ==================================================================
#           Plotting Part d) Cross-Validation MSE
# ==================================================================

# Calculate CV MSE for degrees 1 through 5
degrees = range(1, 6)
cv_mses = [cv_score_degree(x, y, degree=d, k=5, lam=0.0) for d in degrees]

# Extract the Bootstrap Total MSE from Part c results for comparison
bootstrap_total_mse = [r["mse"] for r in results]

plt.figure(figsize=(8, 5))

# Plot CV MSE
plt.plot(degrees, cv_mses, "D--", color="tab:red", label="5-Fold CV MSE", markersize=6)

# Overlay Bootstrap Total MSE
plt.plot(degrees, bootstrap_total_mse, "^-", color="tab:green", label="Bootstrap Total MSE")

plt.yscale("log")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Cross-Validation vs. Bootstrap MSE")
plt.xticks(degrees)
plt.legend()
plt.grid(True)
plt.show()

# 1. Run bootstrap for degrees 1 through 15
degrees = list(range(1, 16))
results = [bootstrap_bias_variance(x, y, degree=d) for d in degrees]

# 2. Extract arrays
bias_sq = np.array([r["bias_sq"] for r in results])
variance = np.array([r["variance"] for r in results])
total_mse = np.array([r["mse"] for r in results])

# 3. Compute the specific statistics needed for report
opt_deg_idx = np.argmin(total_mse)
optimal_degree = degrees[opt_deg_idx]

# Variance growth from degree 1 to degree 15
var_deg_1 = variance[0]
var_deg_15 = variance[-1]
var_growth_factor = var_deg_15 / var_deg_1

# Print the results to fill into LaTeX
print("=== NUMBERS FOR LATEX SECTION ===")
print(f"Optimal Degree (Min Total MSE) : {optimal_degree}")
print(f"Variance at Degree 1          : {var_deg_1:.4e}")
print(f"Variance at Degree 15         : {var_deg_15:.4e}")
print(f"Variance Growth Factor         : {var_growth_factor:.2f}x")
print(f"Min Bias^2 (at high degree)    : {bias_sq[-1]:.4e}")

cv_5 = [cv_score_degree(x, y, degree=d, k=5) for d in range(1, 16)]
cv_10 = [cv_score_degree(x, y, degree=d, k=10) for d in range(1, 16)]

print(f"k=5  Optimal Degree: {np.argmin(cv_5) + 1}")
print(f"k=10 Optimal Degree: {np.argmin(cv_10) + 1}")