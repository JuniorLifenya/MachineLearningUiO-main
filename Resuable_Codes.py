"""
Week 36 Tools: Resampling, Bootstrap, Cross-Validation
Reusable functions for regression, resampling, and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. DATA GENERATION
# ============================================================

def generate_data(n=40, n_features=1, noise=1.0, seed=42, true_coef=None):
    """
    Generate synthetic regression data.
    
    Parameters
    ----------
    n : int
        Number of samples.
    n_features : int
        Number of features.
    noise : float
        Standard deviation of Gaussian noise.
    seed : int
        Random seed for reproducibility.
    true_coef : array-like or None
        True coefficients. If None, random uniform in [-3, 3].
    
    Returns
    -------
    X : ndarray of shape (n, n_features)
    y : ndarray of shape (n,)
    true_coef : ndarray of shape (n_features,)
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    
    if true_coef is None:
        true_coef = rng.uniform(-3, 3, size=n_features)
    
    y = X @ true_coef + rng.normal(scale=noise, size=n)
    return X, y, true_coef


def generate_polynomial_data(n=40, degree=3, noise=0.5, seed=42):
    """
    Generate polynomial regression data.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3, 3, size=n))
    X = np.vander(x, degree + 1, increasing=True)
    true_coef = rng.uniform(-1, 1, size=degree + 1)
    y = X @ true_coef + rng.normal(scale=noise, size=n)
    return X, y, true_coef


# ============================================================
# 2. MODEL FITTING
# ============================================================

def ols_fit(X, y):
    """
    Ordinary Least Squares via normal equation.
    Returns coefficients theta.
    """
    # theta = (X^T X)^{-1} X^T y
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def ridge_fit(X, y, lam=1.0):
    """
    Ridge regression via normal equation.
    theta = (X^T X + lambda I)^{-1} X^T y
    """
    n_features = X.shape[1]
    I = np.eye(n_features)
    return np.linalg.pinv(X.T @ X + lam * I) @ X.T @ y


def predict(X, theta):
    """Predict y from X and theta."""
    return X @ theta


def center_matrix(X):
    """Center each column of X (subtract column mean)."""
    return X - X.mean(axis=0)


def center_vector(y):
    """Center y (subtract mean)."""
    return y - y.mean()


# ============================================================
# 3. EVALUATION METRICS
# ============================================================

def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


def r2(y_true, y_pred):
    """R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def evaluate(y_true, y_pred, name="Model"):
    """Print and return multiple metrics."""
    metrics = {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
    print(f"{name}: MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    return metrics


# ============================================================
# 4. TRAIN/TEST SPLIT
# ============================================================

def split_data(X, y, test_size=0.3, seed=42):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=seed)


# ============================================================
# 5. BOOTSTRAP
# ============================================================

def bootstrap_sample(X, y, rng=None):
    """
    Draw one bootstrap sample (with replacement).
    Returns X_boot, y_boot.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(y)
    idx = rng.integers(0, n, size=n)
    return X[idx], y[idx]


def bootstrap_statistic(X, y, statistic_fn, n_bootstrap=1000, seed=42):
    """
    Compute a statistic on B bootstrap samples.
    
    Parameters
    ----------
    X, y : data
    statistic_fn : callable(X, y) -> float or ndarray
        The statistic to compute (e.g., mean, OLS coefficients).
    n_bootstrap : int
        Number of bootstrap samples.
    
    Returns
    -------
    boot_stats : ndarray of shape (n_bootstrap, ...)
    """
    rng = np.random.default_rng(seed)
    boot_stats = []
    for _ in range(n_bootstrap):
        X_b, y_b = bootstrap_sample(X, y, rng)
        boot_stats.append(statistic_fn(X_b, y_b))
    return np.array(boot_stats)


def bootstrap_mean_ci(data, n_bootstrap=1000, alpha=0.05, seed=42):
    """
    Bootstrap confidence interval for the mean.
    Returns (lower, upper) percentile interval.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means.append(data[idx].mean())
    boot_means = np.array(boot_means)
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper, boot_means


def bootstrap_ols_coefficients(X, y, n_bootstrap=1000, seed=42):
    """
    Bootstrap the OLS coefficients.
    Returns array of shape (n_bootstrap, n_features).
    """
    return bootstrap_statistic(X, y, ols_fit, n_bootstrap, seed)


def bootstrap_ridge_coefficients(X, y, lam=1.0, n_bootstrap=1000, seed=42):
    """Bootstrap Ridge coefficients."""
    fn = lambda X_b, y_b: ridge_fit(X_b, y_b, lam)
    return bootstrap_statistic(X, y, fn, n_bootstrap, seed)


# ============================================================
# 6. CROSS-VALIDATION
# ============================================================

def cross_validate(X, y, model_fn, k=5, seed=42, return_predictions=False):
    """
    K-fold cross-validation.
    
    Parameters
    ----------
    X, y : data
    model_fn : callable(X_train, y_train) -> theta
        Function that fits a model and returns parameters.
    k : int
        Number of folds.
    return_predictions : bool
        If True, return out-of-fold predictions.
    
    Returns
    -------
    cv_errors : list of float
        Validation errors for each fold.
    cv_predictions : ndarray (if return_predictions)
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    cv_errors = []
    oof_preds = np.zeros(len(y))
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        theta = model_fn(X_train, y_train)
        y_pred = predict(X_val, theta)
        
        cv_errors.append(mse(y_val, y_pred))
        oof_preds[val_idx] = y_pred
    
    if return_predictions:
        return cv_errors, oof_preds
    return cv_errors


def cv_summary(cv_errors, name="Model"):
    """Print summary of cross-validation errors."""
    cv_errors = np.array(cv_errors)
    print(f"{name} CV: mean={cv_errors.mean():.4f}, std={cv_errors.std():.4f}, "
          f"min={cv_errors.min():.4f}, max={cv_errors.max():.4f}")
    return cv_errors.mean(), cv_errors.std()


def compare_models_cv(X, y, models_dict, k=5, seed=42):
    """
    Compare multiple models via cross-validation.
    
    Parameters
    ----------
    models_dict : dict
        {name: model_fn} where model_fn(X_train, y_train) -> theta
    """
    results = {}
    for name, model_fn in models_dict.items():
        cv_errors = cross_validate(X, y, model_fn, k=k, seed=seed)
        mean_err, std_err = cv_summary(cv_errors, name)
        results[name] = {"cv_errors": cv_errors, "mean": mean_err, "std": std_err}
    return results


# ============================================================
# 7. LEARNING CURVES
# ============================================================

def learning_curve(X, y, model_fn, train_sizes=None, k=5, seed=42):
    """
    Compute learning curve: training error and validation error
    as a function of training set size.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 0.9, 9)
    
    n = len(y)
    train_errors = []
    val_errors = []
    
    for frac in train_sizes:
        n_train = int(frac * n)
        fold_train_errors = []
        fold_val_errors = []
        
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        for train_idx, val_idx in kf.split(X):
            # Use only a subset of the training fold
            train_idx_subset = train_idx[:n_train]
            X_train, y_train = X[train_idx_subset], y[train_idx_subset]
            X_val, y_val = X[val_idx], y[val_idx]
            
            theta = model_fn(X_train, y_train)
            fold_train_errors.append(mse(y_train, predict(X_train, theta)))
            fold_val_errors.append(mse(y_val, predict(X_val, theta)))
        
        train_errors.append(np.mean(fold_train_errors))
        val_errors.append(np.mean(fold_val_errors))
    
    return np.array(train_sizes) * n, np.array(train_errors), np.array(val_errors)


def plot_learning_curve(X, y, model_fn, model_name="Model", k=5, seed=42):
    """Plot learning curve."""
    sizes, train_err, val_err = learning_curve(X, y, model_fn, k=k, seed=seed)
    
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_err, 'o-', label='Training error')
    plt.plot(sizes, val_err, 's-', label='Validation error')
    plt.xlabel('Training set size')
    plt.ylabel('MSE')
    plt.title(f'Learning Curve: {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. BIAS-VARIANCE DECOMPOSITION
# ============================================================

def bias_variance_decomposition(X_train, y_train, X_test, y_test,
                                 model_fn, n_simulations=100, seed=42):
    """
    Estimate bias^2, variance, and noise for a model.
    
    Returns
    -------
    bias_sq : float
    variance : float
    noise : float
    """
    rng = np.random.default_rng(seed)
    n_test = len(y_test)
    predictions = np.zeros((n_simulations, n_test))
    
    for i in range(n_simulations):
        # Bootstrap the training data
        idx = rng.integers(0, len(y_train), size=len(y_train))
        X_b, y_b = X_train[idx], y_train[idx]
        theta = model_fn(X_b, y_b)
        predictions[i] = predict(X_test, theta)
    
    # Expected prediction
    expected_pred = predictions.mean(axis=0)
    
    # Bias^2
    bias_sq = np.mean((expected_pred - y_test) ** 2)
    
    # Variance
    variance = np.mean(predictions.var(axis=0))
    
    # Noise (irreducible error) - approximate
    noise = np.var(y_test - expected_pred)
    
    return bias_sq, variance, noise


# ============================================================
# 9. DEMONSTRATION: FULL WEEK 36 STORYLINE
# ============================================================

def demo_week36():
    """Run the full Week 36 demonstration."""
    print("=" * 60)
    print("WEEK 36: RESAMPLING, BOOTSTRAP, CROSS-VALIDATION")
    print("=" * 60)
    
    # --- Step 1: Generate data ---
    print("\n--- Step 1: Generate data ---")
    X, y, true_coef = generate_data(n=40, n_features=1, noise=1.0, seed=42)
    print(f"True coefficients: {true_coef}")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # --- Step 2: Train/test split ---
    print("\n--- Step 2: Train/test split ---")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, seed=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # --- Step 3: Fit OLS on training data ---
    print("\n--- Step 3: Fit OLS ---")
    theta_ols = ols_fit(X_train, y_train)
    print(f"OLS coefficients: {theta_ols}")
    
    y_train_pred = predict(X_train, theta_ols)
    y_test_pred = predict(X_test, theta_ols)
    
    evaluate(y_train, y_train_pred, "OLS Train")
    evaluate(y_test, y_test_pred, "OLS Test")
    
    # --- Step 4: Bootstrap the OLS coefficients ---
    print("\n--- Step 4: Bootstrap OLS coefficients ---")
    boot_coefs = bootstrap_ols_coefficients(X_train, y_train, n_bootstrap=1000)
    print(f"Bootstrap mean: {boot_coefs.mean(axis=0)}")
    print(f"Bootstrap std:  {boot_coefs.std(axis=0)}")
    
    # 95% CI
    lower = np.percentile(boot_coefs, 2.5, axis=0)
    upper = np.percentile(boot_coefs, 97.5, axis=0)
    print(f"95% CI lower: {lower}")
    print(f"95% CI upper: {upper}")
    
    # --- Step 5: Cross-validation ---
    print("\n--- Step 5: Cross-validation ---")
    cv_errors = cross_validate(X, y, ols_fit, k=5)
    cv_summary(cv_errors, "OLS")
    
    # --- Step 6: Compare OLS vs Ridge ---
    print("\n--- Step 6: Compare OLS vs Ridge ---")
    models = {
        "OLS": ols_fit,
        "Ridge (lambda=1)": lambda X, y: ridge_fit(X, y, lam=1.0),
        "Ridge (lambda=10)": lambda X, y: ridge_fit(X, y, lam=10.0),
    }
    results = compare_models_cv(X, y, models, k=5)
    
    # --- Step 7: Learning curve ---
    print("\n--- Step 7: Learning curve ---")
    plot_learning_curve(X, y, ols_fit, "OLS", k=5)
    
    # --- Step 8: Bias-variance decomposition ---
    print("\n--- Step 8: Bias-variance decomposition ---")
    bias_sq, var, noise = bias_variance_decomposition(
        X_train, y_train, X_test, y_test, ols_fit
    )
    print(f"Bias^2: {bias_sq:.4f}")
    print(f"Variance: {var:.4f}")
    print(f"Noise: {noise:.4f}")
    print(f"Total: {bias_sq + var + noise:.4f}")


if __name__ == "__main__":
    demo_week36()