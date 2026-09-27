import numpy as np
import matplotlib.pyplot as plt 

# The exercise own starting point
# Runge's function is defined on [-1,1] with 
# Additive Gaussian noise, and pol design matrix



# ============================================================
# 1. THE FUNCTION AND DESIGN MATRIX
# ============================================================

def runge(x):
    """Runge's function: f(x) = 1 / (1 + 25 x²)."""
    return 1.0 / (1.0 + 25.0 * x ** 2)


def design_matrix(x, degree, intercept=True):
    """
    Polynomial design matrix.

    Columns are [1, x, x², ..., x^degree] if intercept=True,
    or [x, x², ..., x^degree] if intercept=False.

    Parameters
    ----------
    x : ndarray of shape (n,)
    degree : int
    intercept : bool

    Returns
    -------
    X : ndarray of shape (n, degree + 1) or (n, degree)
    """
    x = np.asarray(x).ravel()
    start = 0 if intercept else 1
    return np.vstack([x ** p for p in range(start, degree + 1)]).T


def generate_data(n=100, sigma=0.1, seed=2026, x_min=-1.0, x_max=1.0):
    """
    Sample Runge's function with additive Gaussian noise.

    Parameters
    ----------
    n : int
        Number of samples.
    sigma : float
        Noise standard deviation.
    seed : int
    x_min, x_max : float
        Domain boundaries. Defaults to [-1, 1].

    Returns
    -------
    x : ndarray of shape (n,)
    y : ndarray of shape (n,)
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
plt.scatter(x,y, s = 12, color = "orange",label =" data, $\sigma = {sigma}$")
plt.xlabel("x"); plt.ylabel("y");plt.legend(frameon = True)
plt.show()

