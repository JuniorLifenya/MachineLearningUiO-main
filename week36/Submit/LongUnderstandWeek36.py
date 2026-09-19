import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# ==============================================================
# Exercise 3
# Bias-variance tradeoff using bootstrap
# ==============================================================

# Using a random seed makes the pseudo-random data reproducible.
np.random.seed(2026)

n = 40  # Number of observations / samples

# x initially has shape (40,)
# reshape(-1, 1) gives sklearn the expected feature-matrix shape:
#
#       X.shape = (n_samples, n_features)
#
# Here:
#
#       X.shape = (40, 1)
#
x = np.linspace(-3, 3, n).reshape(-1, 1)


# --------------------------------------------------------------
# TRUE UNDERLYING FUNCTION
# --------------------------------------------------------------
#
# Since this is synthetic data, we actually know the true function:
#
#       y = f(x) + noise
#
# This is extremely useful for understanding bias-variance properly.
#

def true_function(x):
    return (
        np.exp(-x**2)
        + 1.5 * np.exp(-(x - 2)**2)
    )


noise_std = 0.1

y = (
    true_function(x)
    + np.random.normal(0, noise_std, x.shape)
)

# sklearn wants a 1D target:
#
#       y.shape = (40,)
#
y = y.ravel()

print("x shape:", x.shape)
print("y shape:", y.shape)


# ==============================================================
# 1. Split into training and test data
# ==============================================================

# 40 observations:
#
#       32 training points
#        8 test points
#
# train_test_split preserves the x-y pairing.
#
# If x_17 goes into the training set,
# the corresponding y_17 also goes into the training set.

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=2026
)

print("\nx_train:", x_train.shape)  # (32, 1)
print("x_test :", x_test.shape)     # (8, 1)
print("y_train:", y_train.shape)    # (32,)
print("y_test :", y_test.shape)     # (8,)


# --------------------------------------------------------------
# KEY IDEA
# --------------------------------------------------------------
#
# The test set is not "another kind" of data.
#
# It comes from the same underlying process.
#
# The important distinction is:
#
#       TRAINING DATA:
#           allowed to influence the fitted model
#
#       TEST DATA:
#           kept unseen while fitting
#
# so that it can act as an approximate measurement of
# generalization to unseen observations.


# Visualize the split

plt.scatter(x_train, y_train, label="Training data")
plt.scatter(x_test, y_test, label="Test data")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# ==============================================================
# 2. First understand ONE model: polynomial degree = 3
# ==============================================================

degree = 3

model = make_pipeline(
    PolynomialFeatures(
        degree=degree,
        include_bias=True
    ),
    LinearRegression(
        fit_intercept=False
    )
)

model.fit(x_train, y_train)

# Prediction at the eight test positions:
#
#       y_pred.shape = (8,)
#
y_pred = model.predict(x_test)

print("\nOne fitted degree-3 model:")
print("Prediction shape:", y_pred.shape)


# ==============================================================
# 3. Bootstrap
# ==============================================================

# Now we want MANY plausible fitted models.
#
# We only observed one training dataset.
#
# Bootstrap asks:
#
# "What if the observed training sample had been slightly different?"
#
# We approximate that by repeatedly resampling the TRAINING data
# WITH REPLACEMENT.


B = 100  # Number of bootstrap datasets


def bootstrap_predictions(degree):

    # Rows    = test observations
    # Columns = bootstrap-fitted models
    #
    # Shape:
    #
    #           (8 test points, 100 models)
    #
    y_pred_boot = np.zeros((len(y_test), B))

    for b in range(B):

        # ------------------------------------------------------
        # Create bootstrap training dataset
        # ------------------------------------------------------
        #
        # Sample 32 observations FROM the 32 original training
        # observations, WITH replacement.
        #
        # Therefore:
        # - some training observations appear several times
        # - some do not appear at all
        #

        x_boot, y_boot = resample(
            x_train,
            y_train,
            replace=True,
            n_samples=len(x_train),
            random_state=2026 + b
        )

        # ------------------------------------------------------
        # Fit polynomial model to this bootstrap dataset
        # ------------------------------------------------------

        model = make_pipeline(
            PolynomialFeatures(
                degree=degree,
                include_bias=True
            ),
            LinearRegression(
                fit_intercept=False
            )
        )

        model.fit(x_boot, y_boot)

        # ------------------------------------------------------
        # IMPORTANT:
        #
        # Every model is evaluated at the SAME x_test points.
        #
        # Therefore column b contains:
        #
        #   predictions of bootstrap model b
        #
        # at all eight fixed test locations.
        # ------------------------------------------------------

        y_pred_boot[:, b] = model.predict(x_test)

    return y_pred_boot


# Test it

pred_deg3 = bootstrap_predictions(degree=3)

print(
    "\nBootstrap prediction matrix shape:",
    pred_deg3.shape
)

# Expected:
#
#       (8, 100)


# ==============================================================
# 4. Understand the prediction matrix
# ==============================================================

# This matrix is VERY important.
#
# Conceptually:
#
#                     bootstrap model
#
#                  1      2      3       ...    100
#
# test point 1    y~11   y~12   y~13     ...   y~1,100
# test point 2    y~21   y~22   y~23     ...   y~2,100
# ...
# test point 8    y~81   y~82   y~83     ...   y~8,100
#
#
# Each COLUMN:
#     one fitted model
#
# Each ROW:
#     what 100 slightly different fitted models predict
#     at the same test location
#
#
# This is essentially the numerical object from which
# we estimate bias and variance.


# ==============================================================
# 5. Mean prediction
# ==============================================================

# Average over the bootstrap models.
#
# axis=1 means:
#
#       average across the columns
#
# so we get one average prediction per test point.

mean_prediction = np.mean(
    pred_deg3,
    axis=1
)

print("\nMean prediction shape:", mean_prediction.shape)

# Shape:
#
#       (8,)


# ==============================================================
# 6. Test error
# ==============================================================

# For every:
#
#       test point i
#       bootstrap model b
#
# calculate:
#
#       (y_i - y~_i,b)^2
#
# Broadcasting:
#
#       y_test[:, np.newaxis] -> shape (8,1)
#
#       pred_deg3             -> shape (8,100)
#
# Result:
#
#       error matrix          -> shape (8,100)

test_error = np.mean(
    (
        y_test[:, np.newaxis]
        - pred_deg3
    )**2
)

print("\nTest error:", test_error)


# ==============================================================
# 7. Variance
# ==============================================================

# At each test point:
#
# Ask:
#
# "How much do the 100 fitted models disagree?"
#
# np.var(..., axis=1)
#
# gives one variance per test point.
#
# Then average across the eight test points.

variance = np.mean(
    np.var(
        pred_deg3,
        axis=1
    )
)

print("Variance:", variance)


# ==============================================================
# 8. Bias squared
# ==============================================================

# --------------------------------------------------------------
# IMPORTANT STATISTICAL DETAIL
# --------------------------------------------------------------
#
# Bias is theoretically defined relative to the TRUE function:
#
#       Bias(x)
#       =
#       E[ y~(x) ] - f(x)
#
#
# Therefore:
#
#       Bias^2
#       =
#       ( E[y~(x)] - f(x) )^2
#
#
# Since this exercise uses synthetic data, we KNOW f(x).
# That means we can calculate a much cleaner estimate of bias.


f_test = true_function(x_test).ravel()

bias_squared = np.mean(
    (
        mean_prediction
        - f_test
    )**2
)

print("Bias^2:", bias_squared)


# ==============================================================
# 9. Irreducible noise
# ==============================================================

# The data were generated as:
#
#       y = f(x) + epsilon
#
# where
#
#       epsilon ~ N(0, sigma^2)
#
# Here:
#
#       sigma = 0.1
#
# therefore:
#
#       sigma^2 = 0.01

noise_variance = noise_std**2

print("Noise variance:", noise_variance)


# ==============================================================
# Bias-variance decomposition
# ==============================================================

print(
    "Bias^2 + Variance + Noise:",
    bias_squared + variance + noise_variance
)

print(
    "Observed bootstrap test error:",
    test_error
)


# We should NOT expect exact equality numerically because:
#
# 1. only 40 total observations
# 2. only 8 test points
# 3. only 100 bootstrap samples
# 4. bootstrap only approximates repeated independent datasets
# 5. the test targets themselves contain random noise
#
# But the quantities should tell the same statistical story.


# ==============================================================
# Exercise 3.1
# Bias-variance tradeoff for degrees 0-13
# ==============================================================

degrees = np.arange(0, 14)

errors = np.zeros(len(degrees))
biases_squared = np.zeros(len(degrees))
variances = np.zeros(len(degrees))


for i, degree in enumerate(degrees):

    # ----------------------------------------------------------
    # Generate all bootstrap predictions ONCE for this degree
    # ----------------------------------------------------------

    y_pred_boot = bootstrap_predictions(degree)

    # ----------------------------------------------------------
    # Mean model prediction
    # ----------------------------------------------------------

    mean_prediction = np.mean(
        y_pred_boot,
        axis=1
    )

    # ----------------------------------------------------------
    # Test error
    #
    # Compare every bootstrap model against observed test y.
    # ----------------------------------------------------------

    errors[i] = np.mean(
        (
            y_test[:, np.newaxis]
            - y_pred_boot
        )**2
    )

    # ----------------------------------------------------------
    # Squared bias
    #
    # Compare average model against TRUE noiseless function.
    # ----------------------------------------------------------

    biases_squared[i] = np.mean(
        (
            mean_prediction
            - f_test
        )**2
    )

    # ----------------------------------------------------------
    # Variance
    #
    # How strongly do bootstrap models disagree?
    # ----------------------------------------------------------

    variances[i] = np.mean(
        np.var(
            y_pred_boot,
            axis=1
        )
    )


# ==============================================================
# Plot bias-variance tradeoff
# ==============================================================

plt.plot(
    degrees,
    errors,
    marker="o",
    label="Test error"
)

plt.plot(
    degrees,
    biases_squared,
    marker="o",
    label="Bias$^2$"
)

plt.plot(
    degrees,
    variances,
    marker="o",
    label="Variance"
)

plt.xlabel("Polynomial degree")
plt.ylabel("Error")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.show()


# ==============================================================
# Optional: include irreducible noise in decomposition
# ==============================================================

plt.plot(
    degrees,
    errors,
    marker="o",
    label="Observed test error"
)

plt.plot(
    degrees,
    biases_squared + variances + noise_variance,
    marker="o",
    label=r"Bias$^2$ + Variance + $\sigma^2$"
)

plt.xlabel("Polynomial degree")
plt.ylabel("Error")
plt.title("Bias-Variance Decomposition")
plt.legend()
plt.show()


# ==============================================================
# 10. Look at one test point directly
# ==============================================================

pred = bootstrap_predictions(degree=3)

# Select test point zero:
#
#       pred[0, :]
#
# These are 100 predictions at the SAME x coordinate,
# produced by 100 different bootstrap-fitted models.

plt.hist(
    pred[0, :],
    bins=15
)

plt.xlabel("Predicted value")
plt.ylabel("Count")
plt.title("Bootstrap predictions at test point 0")
plt.show()


# ==============================================================
# Interpretation of the histogram
# ==============================================================

# Narrow histogram:
#
#       bootstrap models mostly agree
#       -> LOW variance
#
# Wide histogram:
#
#       bootstrap models strongly disagree
#       -> HIGH variance
#
#
# But this says NOTHING by itself about bias.
#
# The entire narrow histogram could be centered at the wrong value.
#
# Example:
#
#       true value = 2
#
# models predict:
#
#       5.00, 5.02, 4.99, 5.01, ...
#
# Then:
#
#       LOW variance
#       HIGH bias
#
#
# This distinction is the heart of the exercise.