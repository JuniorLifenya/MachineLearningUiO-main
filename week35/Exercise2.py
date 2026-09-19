import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# Exercise 3
# ============================================================


# =========== Data Generation ===============================

rng = np.random.default_rng(2026)

x = rng.random((100, 1))

y = 2.0 + 5.0 * x * x + 0.1 * rng.standard_normal((100, 1))

# =========== Data Generation ===============================

# =========== Design Marix making ===========================

X = np.column_stack((
    np.ones(len(x)),
    x[:, 0 ],
    x[:,0]**2
))

# Just to check 
print("Shape of X", X.shape)
print(X[:5])
print("Now The model is y = Xtheta")

# =========== Design Matrix making ===========================

# =========== Own OLS FIT ====================================

theta_own = np.linalg.lstsq(X, y, rcond=None)[0]

print("Own theta:")
print(theta_own)

# =========== Own OLS FIT ====================================

# ================ Compare Scikit ============================

X_sklearn = np.column_stack((
    x[:, 0],
    x[:, 0]**2
))

model = LinearRegression()

model.fit(X_sklearn, y)

print("sklearn intercept:", model.intercept_)
print("sklearn coefficients:", model.coef_)

#  Combine them 

theta_sklearn = np.array([
    model.intercept_[0],
    model.coef_[0, 0],
    model.coef_[0, 1]
])

print("Own parameters:")
print(theta_own.ravel())

print("Sklearn parameters:")
print(theta_sklearn)

print("Agreement:")
print(np.allclose(theta_own.ravel(), theta_sklearn))

# ================ Compare Scikit ============================

# ================ Compute MSE ===============================
#                 3. MSE and R^2

y_tilde = model.predict(X_sklearn)

mse = mean_squared_error(y, y_tilde)
r2 = r2_score(y, y_tilde)

print("MSE =", mse)
print("R^2 =", r2)

# Fitted coefficients are close to the true parameters (2,0,5), 
# despite the added stochastic noise. 
# The MSE is small, indicating that the fitted polynomial lies close to the observed data. 
# The value R^2 ca .993 indicates that approximately 99.3%
# of the variation in the data is explained by the quadratic model.

# ================ Compute MSE =================================

# ================ Effect of Noise =============================
# Increasing the noise amplitude increases the MSE and decreases R^2. 
# The fitted coefficients also fluctuate more strongly around the true values 
# because the observations contain less information about the underlying 
# deterministic function.


# ======================= Plot OLS FIT =========================

noise_levels = [0.1, 0.5, 1.0]

for noise in noise_levels:

    rng_noise = np.random.default_rng(2026)

    y_noise = (
        2.0
        + 5.0 * x**2
        + noise * rng_noise.standard_normal((100, 1))
    )

    model_noise = LinearRegression()
    model_noise.fit(X_sklearn, y_noise)

    y_pred_noise = model_noise.predict(X_sklearn)

    mse_noise = mean_squared_error(y_noise, y_pred_noise)
    r2_noise = r2_score(y_noise, y_pred_noise)

    print(
        f"noise={noise:3.1f}  "
        f"MSE={mse_noise:.4f}  "
        f"R2={r2_noise:.4f}"
    )

    # Plot each noisy dataset to see the variance!
    plt.scatter(x, y_noise, alpha=0.4, label=f"noise {noise}")

# ========== OUTSIDE LOOP =====================================

# Create the smooth line for your original prediction
x_plot = np.linspace(0, 1, 300).reshape(-1, 1)

X_plot = np.column_stack((
    x_plot[:, 0],
    x_plot[:, 0]**2
))

y_plot = model.predict(X_plot)

plt.plot(x_plot, y_plot, color="black", linewidth=2, label="original quadratic OLS")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
