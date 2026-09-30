import numpy as np

# ==================================================================
#                   Part h)-Stochastic Grad Descent
#                     Same optimizers as before,
#               but each step uses a random mini-bach instead
#                   No full training set used here.
# ==================================================================

def sgd(grad_fn,X,y,theta0,n_epochs, batch_size,eta,
        optimizer ="adam", **kw):
    n = len(y)
    theta = theta0.astype(float).copy()

    # Optimizer State
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    s = np.zeros_like(theta)
    t = 0

    rng = np.random.default_rng(0)

    for epoch in range(n_epochs):
        idx = rng.permutation(n)
        X_sh,y_sh = X[idx], y[idx]
        for start in range(0,n,batch_size):
            Xb = X_sh[start:start + batch_size]
            yb = y_sh[start:start + batch_size]
            g = np.asarray(grad_fn(theta,Xb,yb,**kw))
            t +=1

            if optimizer == "sgd":
                theta -= eta*g
            elif optimizer == "adam":
                m = 0.9*m + 0.1*g
                v = 0.999*v + 0.001*g**2
                m_hat = m / (1-0.9**t)
                v_hat = v / (1-0.999**t)
                theta = theta -eta*m_hat / (np.sqrt(v_hat) + 1e-8)
            # extend for others as needed
        return theta

# mini-batch size (16, 32, 128, full), number of epochs, learning-rate schedule (constant vs decaying).
# Discuss: smaller batches are noisier but escape sharp minima; 
# larger batches are more stable but slower per epoch. 
# Cost per epoch is higher for SGD because of the Python loop, so SGD only wins for very large n.

# ==================================================================
#                   Part i)-Final Model Selection
#                     Use the CV setup from part d) 
#               Pick (degree, lam) for OLS, Ridge, and Lasso,
#                       Compare the three here
# ==================================================================