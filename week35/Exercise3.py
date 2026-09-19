import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ============================================================
# Exercise 3
# ============================================================

rng = np.random.default_rng(2026)

# Generate data
n = 100

x_3 = np.linspace(-3, 3, n).reshape(-1, 1)

y_3 = (
    np.exp(-x_3**2)
    + 1.5 * np.exp(-(x_3 - 2)**2)
    + rng.normal(0, 0.1, x_3.shape)
)


# Lists for storing MSE values
train_mse = []
test_mse = []

degrees = range(1, 16)

for d in degrees:

    x_train, x_test, y_train, y_test = train_test_split(
    x_3,
    y_3,
    test_size=0.2,
    random_state=2026
)

    poly = PolynomialFeatures(
        degree=d,
        include_bias=True
    )

    X_train = poly.fit_transform(x_train)
    X_test = poly.transform(x_test)

    model = LinearRegression(
        fit_intercept=False
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse.append(
        mean_squared_error(y_train, y_train_pred)
    )

    test_mse.append(
        mean_squared_error(y_test, y_test_pred)
    )

# ============================================================
# Find best polynomial degree
# ============================================================

best_index = np.argmin(test_mse)

best_degree = list(degrees)[best_index]

print("Best polynomial degree:", best_degree)
print("Smallest test MSE:", test_mse[best_index])


# ============================================================
# Plot
# ============================================================

plt.plot(
    degrees,
    train_mse,
    "o-",
    label="Training MSE"
)

plt.plot(
    degrees,
    test_mse,
    "s-",
    label="Test MSE"
)

plt.axvline(
    best_degree,
    linestyle="--",
    label=f"Best degree = {best_degree}"
)

plt.xlabel("Polynomial degree")
plt.ylabel("Mean Squared Error")

plt.yscale("log")

plt.xticks(list(degrees))

plt.legend()

plt.tight_layout()

plt.show()

# As the polynomial degree increases, 
# the training MSE decreases because the higher-degree models have greater flexibility 
# and contain the lower-degree models as special cases. 
# The test MSE initially decreases as the model captures the structure of the underlying function. 
# Beyond an optimal degree, however, the test error increases while the training error continues to decrease.
# This is evidence of overfitting: the model begins fitting stochastic fluctuations in the training data rather 
# than only the underlying signal. The polynomial degree corresponding to the minimum test MSE therefore 
# provides the best model among those tested.