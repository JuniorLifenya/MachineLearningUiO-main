import numpy as np
import matplotlib.pyplot as plt 

# The exercise own starting point
# Runge's function is defined on [-1,1] with 
# Additive Gaussian noise, and pol design matrix

# ============================================================
# 1. THE FUNCTION AND DESIGN MATRIX
# ============================================================

def runge(x):
    """
    Runge's function: f(x) = 1 / (1 + 25 x²).
    """
    return 1.0 / (1.0 + 25.0 * x ** 2)


def design_matrix(x, degree, intercept=True):
    """
    Polynomial design matrix.

    Columns are [1, x, x², ..., x^degree] if intercept=True,
    or [x, x², ..., x^degree] if intercept=False.

    """
    x = np.asarray(x).ravel()
    start = 0 if intercept else 1
    return np.vstack([x ** p for p in range(start, degree + 1)]).T


def generate_data(n=100, sigma=0.1, seed=2026, x_min=-1.0, x_max=1.0):
    """
    Sample Runge's function with additive Gaussian noise.

    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(x_min, x_max, n))
    y = runge(x) + rng.normal(0.0, sigma, n)
    return x, y


rng = np.random.default_rng(2026)
n = 100
sigma = 0.1
x = np.sort(rng.uniform(-1,1,n))
y = runge(x) + rng.normal(0, sigma,n)

xx = np.linspace (-1,1, 400)
plt.plot(xx, runge(xx), color = "green", label= "Runge function fitting test")
plt.scatter(x,y, s = 12, color = "orange",label = r" data, $\sigma = {sigma}$")
plt.xlabel("x"); plt.ylabel("y");plt.legend(frameon = True)
plt.show()


# ============================================================
# Fitting Codes 
# ============================================================

def fit_ols_SVD(X,y,rcond = 1e-15):
    """
    OLS via SVD: theta = V diag(1/s_i) U^T y.
    Singular values below rcond * s_max are zeroed.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    return Vt.T @ (s_inv * (U.T @ y))

def fit_ridge(X,y,lam):
    """
    Ridge via SVD: theta = V diag(s_i / (s_i^2 + lam)) U^T y.
    This is the shrinkage view.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    shrink = s / (s ** 2 + lam)
    return Vt.T @ (shrink * (U.T @ y))


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    n = 100
    sigma = 0.1
    x = np.sort(rng.uniform(-1,1,n))
    y = runge(x) + rng.normal(0, sigma,n)

    xx = np.linspace (-1,1, 400)
    plt.plot(xx, runge(xx), color = "green", label= "Runge function fitting test")
    # Fixed string formatting with 'r'
    plt.scatter(x,y, s = 12, color = "orange",label = r" data, $\sigma = {sigma}$")
    plt.xlabel("x"); plt.ylabel("y");plt.legend(frameon = True)
    plt.show()

    X = design_matrix(x, degree=18, intercept=True)
    cond_xtx = np.linalg.cond(X.T @ X)
    exponent = np.log10(cond_xtx)
    print(f"Condition number order of magnitude: 10^{exponent:.1f}")