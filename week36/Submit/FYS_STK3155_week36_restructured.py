import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# ============================================================
# GLOBAL SETTINGS
# ============================================================

SEED = 2026


# ============================================================
# Helper functions
# ============================================================

def make_data(n=40, seed=SEED, noise_std=0.1):
    """
    Exercise-3 data:
        y = exp(-x^2) + 1.5 exp(-(x-2)^2) + epsilon
        epsilon ~ N(0, noise_std^2)

    Returns
    -------
    x : shape (n, 1)
    y : shape (n,)
    """
    # RandomState reproduces the traditional np.random.seed(seed) sequence.
    rng = np.random.RandomState(seed)

    x = np.linspace(-3.0, 3.0, n).reshape(-1, 1)

    f = (
        np.exp(-x**2)
        + 1.5 * np.exp(-(x - 2.0)**2)
    )

    y = f + rng.normal(
        loc=0.0,
        scale=noise_std,
        size=x.shape
    )

    return x, y.ravel()


def make_bootstrap_indices(n_train, B=100, seed=SEED):
    """
    Precompute B bootstrap samples.

    Each row contains n_train indices sampled WITH replacement.
    Shape: (B, n_train)
    """
    rng = np.random.default_rng(seed)

    return rng.integers(
        low=0,
        high=n_train,
        size=(B, n_train)
    )


def bootstrap_predictions(
    model_factory,
    x_train,
    y_train,
    x_test,
    bootstrap_indices
):
    """
    Fit one model for every bootstrap resample of the training set.

    Returns
    -------
    predictions : shape (n_test, B)

    Rows    = test observations
    Columns = bootstrap-fitted models
    """
    B = bootstrap_indices.shape[0]

    predictions = np.empty(
        (len(x_test), B)
    )

    for b, idx in enumerate(bootstrap_indices):
        model = model_factory()

        model.fit(
            x_train[idx],
            y_train[idx]
        )

        predictions[:, b] = model.predict(x_test).ravel()

    return predictions


def bias_variance_from_predictions(predictions, y_test):
    """
    Numerical quantities used in the course.

    IMPORTANT:
    The quantity called 'bias2' below is evaluated against noisy y_test,
    not the unknown true f(x). It therefore contains irreducible noise.

    predictions shape = (n_test, B)
    """
    y_test = np.asarray(y_test).ravel()

    mean_prediction = predictions.mean(axis=1)

    # Mean squared prediction error over:
    #   test points AND bootstrap-fitted models.
    mse = np.mean(
        (predictions - y_test[:, None])**2
    )

    # Course's numerical "bias^2":
    # compare noisy test target to the average prediction.
    bias2 = np.mean(
        (y_test - mean_prediction)**2
    )

    # For each test point:
    # variance across bootstrap models.
    # Then average over test points.
    variance = np.mean(
        np.var(
            predictions,
            axis=1,
            ddof=0
        )
    )

    return mse, bias2, variance


def bootstrap_stats(
    model_factory,
    x_train,
    y_train,
    x_test,
    y_test,
    bootstrap_indices
):
    """
    Convenience wrapper:
        bootstrap fits -> predictions -> MSE/bias^2/variance.
    """
    predictions = bootstrap_predictions(
        model_factory,
        x_train,
        y_train,
        x_test,
        bootstrap_indices
    )

    return bias_variance_from_predictions(
        predictions,
        y_test
    )


def kfold_cv_mse(
    model,
    X,
    y,
    k=5,
    shuffle=True,
    seed=SEED
):
    """
    Manual k-fold cross-validation.

    The model can be a Pipeline. Because the model is cloned and fitted
    inside each fold, preprocessing inside the Pipeline is also fitted
    using ONLY the training portion of that fold.

    Returns
    -------
    mean_mse, std_mse, fold_mses
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n = len(y)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    folds = np.array_split(indices, k)

    fold_mses = []

    for i in range(k):
        val_idx = folds[i]

        train_idx = np.concatenate(
            [
                folds[j]
                for j in range(k)
                if j != i
            ]
        )

        fitted_model = clone(model)

        fitted_model.fit(
            X[train_idx],
            y[train_idx]
        )

        pred = fitted_model.predict(
            X[val_idx]
        ).ravel()

        mse = np.mean(
            (y[val_idx] - pred)**2
        )

        fold_mses.append(mse)

    fold_mses = np.asarray(fold_mses)

    return (
        fold_mses.mean(),
        fold_mses.std(ddof=1) if k > 1 else 0.0,
        fold_mses
    )


# ============================================================
# Exercise 3
# 1) Split data 80/20.
# 2) Degrees 0-13.
# 3) Bootstrap training data B=100 times.
# 4) Estimate test error, measured bias^2, and variance.
# ============================================================

def exercise_3_degree_sweep(
    n=40,
    degrees=np.arange(14),
    B=100,
    seed=SEED,
    make_plot=True
):
    x, y = make_data(
        n=n,
        seed=seed
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=seed
    )

    # Same bootstrap index sets are reused for every polynomial degree.
    # This makes the degree comparison cleaner.
    boot_idx = make_bootstrap_indices(
        n_train=len(y_train),
        B=B,
        seed=seed
    )

    errors = np.zeros(len(degrees))
    biases = np.zeros(len(degrees))
    variances = np.zeros(len(degrees))

    for i, degree in enumerate(degrees):

        def model_factory(degree=degree):
            # include_bias=True creates:
            # degree 0 -> [1]
            # degree 3 -> [1, x, x^2, x^3]
            #
            # Therefore fit_intercept=False avoids a duplicate constant.
            return make_pipeline(
                PolynomialFeatures(
                    degree=degree,
                    include_bias=True
                ),
                LinearRegression(
                    fit_intercept=False
                )
            )

        mse, bias2, var = bootstrap_stats(
            model_factory,
            x_train,
            y_train,
            x_test,
            y_test,
            boot_idx
        )

        errors[i] = mse
        biases[i] = bias2
        variances[i] = var

    if make_plot:
        plt.figure(figsize=(7, 4))

        plt.plot(
            degrees,
            errors,
            "o-",
            label="Test error (MSE)"
        )

        plt.plot(
            degrees,
            biases,
            "s-",
            label="Measured bias²"
        )

        plt.plot(
            degrees,
            variances,
            "^-",
            label="Variance"
        )

        plt.xlabel("Polynomial degree")
        plt.ylabel("Error")
        plt.title(
            f"Bias-variance tradeoff "
            f"(n={n}, B={B})"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "n": n,
        "degrees": np.asarray(degrees),
        "mse": errors,
        "bias2": biases,
        "var": variances,
        "best_degree": int(
            degrees[np.argmin(errors)]
        ),
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "boot_idx": boot_idx
    }


def exercise_3_compare_sample_sizes(
    sample_sizes=(40, 100, 400),
    degrees=np.arange(14),
    B=100,
    seed=SEED
):
    """
    Exercise 3.3:
    repeat the bias-variance study for n=40,100,400.
    """
    results = {}

    plt.figure(figsize=(8, 5))

    for n in sample_sizes:
        result = exercise_3_degree_sweep(
            n=n,
            degrees=degrees,
            B=B,
            seed=seed,
            make_plot=False
        )

        results[n] = result

        plt.plot(
            degrees,
            result["mse"],
            marker="o",
            label=f"n={n}"
        )

        print(
            f"n={n:3d}: "
            f"best degree = "
            f"{result['best_degree']}"
        )

    plt.xlabel("Polynomial degree")
    plt.ylabel("Bootstrap test MSE")
    plt.title("Effect of increasing sample size")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results


def exercise_3_ridge(
    n=40,
    degree=12,
    B=100,
    lambdas=np.logspace(-6, 2, 9),
    seed=SEED
):
    """
    Exercise 3.4:
    fixed high-degree polynomial + Ridge,
    then study bias/variance as lambda changes.
    """
    x, y = make_data(
        n=n,
        seed=seed
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=seed
    )

    boot_idx = make_bootstrap_indices(
        n_train=len(y_train),
        B=B,
        seed=seed
    )

    mse_values = []
    bias_values = []
    var_values = []

    for lam in lambdas:

        def model_factory(lam=lam):
            # Important:
            # scaling lives INSIDE the pipeline,
            # so every bootstrap model fits its own scaler
            # using only that bootstrap training sample.
            return make_pipeline(
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False
                ),
                StandardScaler(),
                Ridge(
                    alpha=lam,
                    fit_intercept=True
                )
            )

        mse, bias2, var = bootstrap_stats(
            model_factory,
            x_train,
            y_train,
            x_test,
            y_test,
            boot_idx
        )

        mse_values.append(mse)
        bias_values.append(bias2)
        var_values.append(var)

    mse_values = np.asarray(mse_values)
    bias_values = np.asarray(bias_values)
    var_values = np.asarray(var_values)

    best_lambda = lambdas[
        np.argmin(mse_values)
    ]

    plt.figure(figsize=(7, 4))
    plt.semilogx(
        lambdas,
        mse_values,
        "o-",
        label="MSE"
    )
    plt.semilogx(
        lambdas,
        bias_values,
        "s-",
        label="Measured bias²"
    )
    plt.semilogx(
        lambdas,
        var_values,
        "^-",
        label="Variance"
    )

    plt.axvline(
        best_lambda,
        linestyle="--",
        label=f"best λ={best_lambda:.2e}"
    )

    plt.xlabel(r"$\lambda$")
    plt.ylabel("Error")
    plt.title(
        f"Ridge tradeoff: degree {degree}, n={n}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(
        "Ridge best lambda:",
        best_lambda
    )

    return {
        "lambda": np.asarray(lambdas),
        "mse": mse_values,
        "bias2": bias_values,
        "var": var_values,
        "best_lambda": best_lambda
    }


# ============================================================
# Exercise 4
# Cross-validation
# ============================================================

def exercise_4_degree_cv(
    n=40,
    degrees=np.arange(14),
    k=5,
    seed=SEED
):
    """
    Exercise 4.2:
    choose polynomial degree with manual k-fold CV.
    """
    x, y = make_data(
        n=n,
        seed=seed
    )

    cv_mean = np.zeros(len(degrees))
    cv_std = np.zeros(len(degrees))

    for i, degree in enumerate(degrees):
        model = make_pipeline(
            PolynomialFeatures(
                degree=degree,
                include_bias=True
            ),
            LinearRegression(
                fit_intercept=False
            )
        )

        mean_mse, std_mse, _ = kfold_cv_mse(
            model,
            x,
            y,
            k=k,
            shuffle=True,
            seed=seed
        )

        cv_mean[i] = mean_mse
        cv_std[i] = std_mse

    best_degree = int(
        degrees[np.argmin(cv_mean)]
    )

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        degrees,
        cv_mean,
        yerr=cv_std,
        marker="o",
        capsize=3
    )
    plt.xlabel("Polynomial degree")
    plt.ylabel(f"{k}-fold CV MSE")
    plt.title(
        f"Cross-validation vs degree (n={n})"
    )
    plt.tight_layout()
    plt.show()

    print(
        f"{k}-fold CV best degree:",
        best_degree
    )

    return {
        "degrees": np.asarray(degrees),
        "mean_mse": cv_mean,
        "std_mse": cv_std,
        "best_degree": best_degree
    }


def make_ridge_cv_data(
    n=100,
    seed=SEED
):
    """
    Exercise 4.3 data:
        y = 3 x^2 + epsilon,
        epsilon ~ N(0,1).

    A random Gaussian x sample is used here.
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n, 1)
    )

    y = (
        3.0 * x.ravel()**2
        + rng.normal(
            loc=0.0,
            scale=1.0,
            size=n
        )
    )

    return x, y


def ridge_cv_curve(
    x,
    y,
    lambdas,
    degree=6,
    k=5,
    seed=SEED
):
    """
    Compute the CV curve for Ridge.

    Polynomial expansion + scaling are INSIDE the Pipeline,
    so both are fitted independently inside each training fold.
    """
    cv_mean = np.zeros(len(lambdas))
    cv_std = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        model = make_pipeline(
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            ),
            StandardScaler(),
            Ridge(
                alpha=lam,
                fit_intercept=True
            )
        )

        mean_mse, std_mse, _ = kfold_cv_mse(
            model,
            x,
            y,
            k=k,
            shuffle=True,
            seed=seed
        )

        cv_mean[i] = mean_mse
        cv_std[i] = std_mse

    best_idx = np.argmin(cv_mean)

    return {
        "lambda": np.asarray(lambdas),
        "mean_mse": cv_mean,
        "std_mse": cv_std,
        "best_lambda": lambdas[best_idx],
        "best_mse": cv_mean[best_idx]
    }


def exercise_4_ridge_lambda(
    n=100,
    degree=6,
    lambdas=np.logspace(-3, 5, 100),
    k=5,
    seed=SEED
):
    """
    Exercise 4.3:
    choose Ridge lambda using k-fold CV.
    """
    x, y = make_ridge_cv_data(
        n=n,
        seed=seed
    )

    result = ridge_cv_curve(
        x,
        y,
        lambdas=lambdas,
        degree=degree,
        k=k,
        seed=seed
    )

    plt.figure(figsize=(7, 4))
    plt.semilogx(
        result["lambda"],
        result["mean_mse"]
    )
    plt.axvline(
        result["best_lambda"],
        linestyle="--",
        label=(
            f"best λ="
            f"{result['best_lambda']:.2e}"
        )
    )
    plt.xlabel(r"$\lambda$")
    plt.ylabel(f"{k}-fold CV MSE")
    plt.title("Ridge hyperparameter selection")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(
        "Best lambda:",
        result["best_lambda"]
    )
    print(
        "Best CV MSE:",
        result["best_mse"]
    )

    return x, y, result


def exercise_4_compare_k(
    x,
    y,
    degree=6,
    lambdas=np.logspace(-3, 5, 100),
    ks=(5, 10, None),
    seed=SEED
):
    """
    Exercise 4.4:
    compare k=5, k=10, and leave-one-out (k=n).
    """
    results = {}

    for k in ks:
        k_actual = len(y) if k is None else k

        result = ridge_cv_curve(
            x,
            y,
            lambdas=lambdas,
            degree=degree,
            k=k_actual,
            seed=seed
        )

        results[k_actual] = result

        print(
            f"k={k_actual:3d}: "
            f"best λ = "
            f"{result['best_lambda']:.3e}"
        )

    return results


# ============================================================
# Exercise 5
# Bootstrap uncertainty for fitted parameters
# ============================================================

def exercise_5_ols(
    n=40,
    degree=3,
    B=1000,
    seed=SEED
):
    """
    Exercise 5.1:
    compare analytical OLS standard errors with bootstrap spread.
    """
    x, y = make_data(
        n=n,
        seed=seed
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=seed
    )

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=True
    )

    X_train = poly.fit_transform(
        x_train
    )

    n_train, p = X_train.shape

    # ---------- Analytical OLS ----------
    ols = LinearRegression(
        fit_intercept=False
    )

    ols.fit(
        X_train,
        y_train
    )

    theta_hat = ols.coef_

    residuals = (
        y_train
        - ols.predict(X_train)
    )

    # Residual variance:
    # sigma_hat^2 = RSS / (n-p)
    sigma2_hat = (
        residuals @ residuals
    ) / (n_train - p)

    XtX = X_train.T @ X_train

    # Avoid explicit np.linalg.inv(...).
    # Solve (X^T X) A = I for A.
    XtX_inv = np.linalg.solve(
        XtX,
        np.eye(p)
    )

    covariance = (
        sigma2_hat * XtX_inv
    )

    se_analytical = np.sqrt(
        np.diag(covariance)
    )

    # ---------- Pairs bootstrap ----------
    boot_idx = make_bootstrap_indices(
        n_train=n_train,
        B=B,
        seed=seed
    )

    theta_boot = np.empty(
        (B, p)
    )

    for b, idx in enumerate(boot_idx):
        model_b = LinearRegression(
            fit_intercept=False
        )

        model_b.fit(
            X_train[idx],
            y_train[idx]
        )

        theta_boot[b] = model_b.coef_

    se_boot = theta_boot.std(
        axis=0,
        ddof=1
    )

    print("OLS coefficients :", theta_hat)
    print("Analytical SE    :", se_analytical)
    print("Bootstrap SE     :", se_boot)

    fig, axes = plt.subplots(
        1,
        p,
        figsize=(14, 3)
    )

    axes = np.atleast_1d(axes)

    for j, ax in enumerate(axes):
        ax.hist(
            theta_boot[:, j],
            bins=30
        )
        ax.axvline(
            theta_hat[j],
            linestyle="--"
        )
        ax.set_title(
            rf"$\theta_{j}$"
        )
        ax.set_xlabel(
            "bootstrap estimate"
        )

    plt.tight_layout()
    plt.show()

    return {
        "theta_hat": theta_hat,
        "se_analytical": se_analytical,
        "se_boot": se_boot,
        "theta_boot": theta_boot,
        "x_train": x_train,
        "y_train": y_train
    }


def exercise_5_lasso(
    x_train,
    y_train,
    degree=3,
    alpha=0.01,
    B=1000,
    seed=SEED
):
    """
    Exercise 5.2:
    bootstrap Lasso coefficients.

    For Lasso:
      - do not penalise the intercept;
      - standardise the polynomial features;
      - inspect selection frequency as well as coefficient spread.

    To keep every bootstrap coefficient expressed in the SAME standardized
    coordinate system, the scaler is fitted once on the original training set.
    """
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )

    Phi_train = poly.fit_transform(
        x_train
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        Phi_train
    )

    n_train, p = X_train_scaled.shape

    boot_idx = make_bootstrap_indices(
        n_train=n_train,
        B=B,
        seed=seed
    )

    theta_boot = np.empty(
        (B, p)
    )

    intercept_boot = np.empty(B)

    for b, idx in enumerate(boot_idx):
        model_b = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=10000
        )

        model_b.fit(
            X_train_scaled[idx],
            y_train[idx]
        )

        theta_boot[b] = model_b.coef_
        intercept_boot[b] = model_b.intercept_

    coefficient_spread = theta_boot.std(
        axis=0,
        ddof=1
    )

    selection_frequency = np.mean(
        np.abs(theta_boot) > 1e-12,
        axis=0
    )

    print(
        "Lasso bootstrap coefficient spread:",
        coefficient_spread
    )

    print(
        "Lasso selection frequency:",
        selection_frequency
    )

    fig, axes = plt.subplots(
        1,
        p,
        figsize=(12, 3)
    )

    axes = np.atleast_1d(axes)

    for j, ax in enumerate(axes):
        ax.hist(
            theta_boot[:, j],
            bins=30
        )
        ax.set_title(
            rf"$\theta_{j+1}$"
        )
        ax.set_xlabel(
            "bootstrap Lasso coefficient"
        )

    plt.tight_layout()
    plt.show()

    return {
        "theta_boot": theta_boot,
        "intercept_boot": intercept_boot,
        "coefficient_spread": coefficient_spread,
        "selection_frequency": selection_frequency,
        "scaler": scaler,
        "poly": poly
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Exercise 3.1 + 3.2
    # --------------------------------------------------------
    ex3_n40 = exercise_3_degree_sweep(
        n=40,
        B=100
    )

    print(
        "Exercise 3: n=40 best degree =",
        ex3_n40["best_degree"]
    )

    # --------------------------------------------------------
    # Exercise 3.3
    # --------------------------------------------------------
    ex3_sizes = exercise_3_compare_sample_sizes(
        sample_sizes=(40, 100, 400),
        B=100
    )

    # --------------------------------------------------------
    # Exercise 3.4
    # --------------------------------------------------------
    ex3_ridge = exercise_3_ridge(
        n=40,
        degree=12,
        B=100
    )

    # --------------------------------------------------------
    # Exercise 4.2
    # --------------------------------------------------------
    ex4_degree = exercise_4_degree_cv(
        n=40,
        k=5
    )

    # --------------------------------------------------------
    # Exercise 4.3
    # --------------------------------------------------------
    x_ridge, y_ridge, ex4_ridge = exercise_4_ridge_lambda(
        n=100,
        degree=6,
        k=5
    )

    # --------------------------------------------------------
    # Exercise 4.4
    # --------------------------------------------------------
    ex4_compare = exercise_4_compare_k(
        x_ridge,
        y_ridge,
        degree=6,
        ks=(5, 10, None)
    )

    # --------------------------------------------------------
    # Exercise 5.1
    # --------------------------------------------------------
    ex5_ols = exercise_5_ols(
        n=40,
        degree=3,
        B=1000
    )

    # --------------------------------------------------------
    # Exercise 5.2
    # --------------------------------------------------------
    ex5_lasso = exercise_5_lasso(
        ex5_ols["x_train"],
        ex5_ols["y_train"],
        degree=3,
        alpha=0.01,
        B=1000
    )
