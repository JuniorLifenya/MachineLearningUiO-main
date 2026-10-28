# Reusing our gradient machinery from project 1,
# We need the following functions and gradients
import numpy as np
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys 
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0,str(project_root))

# ========================================================================
# 1 MSE functions
# ========================================================================

def C_mse(y_targ, y_pred): # The main mse Cost function
    y_targ = np.asarray(y_targ)
    y_pred = np.asarray(y_pred)

    if y_targ.shape != y_pred.shape:
        raise ValueError("y_targ and y_pred have different shapes")
    
    return np.mean((y_targ - y_pred) ** 2) 

def l1_penalty(params):
    return sum(np.sum(np.abs(p)) for p in params)

def l2_penalty(params):
    return sum(float(np.sum(np.asarray(p)**2)) for p in params)

def mse_l1(y_targ,y_pred,params,lam):
    return C_mse(y_targ,y_pred) + lam * l1_penalty(params)

def mse_l2(y_targ,y_pred,params,lam):
    return C_mse(y_targ,y_pred) + lam * l2_penalty(params)

# Here the params will be a list (params = [W1,b1,W2,b2])
#  of Neural network trainable arrays
# The penalties are not on the theta params anymore,
# but the weight matrices W(l) for Neural Networks

# -------------- 1.5 MSE function derivatives -----------------

def dC_dypred_batch(y_targ,y_pred):
    """
        dC/dy_pred for C = (1/n) sum (y_pred - y_targ)^2.
    """
    n = y_pred.size
    return (2.0/n) * (y_pred-y_targ)

def L1_penalty_grad(params, lam):
    """Gradient of the L1 penalty with respect to each array in params.
    Returns a list of arrays, one per parameter, each with the same shape."""
    return [lam * np.sign(np.asarray(p)) for p in params]

def L2_penalty_grad(params, lam):
    return [2.0 * lam * np.asarray(p) for p in params]



# ========================================================================
# 2 Binary cross entropy (log loss)
# ========================================================================

def BC_entropy(y_targ,y_pred,eps = 1e-12):
    y_pred = jnp.clip(y_pred, eps, 1.0-eps)
    return -jnp.mean(
        y_targ*jnp.log(y_pred) 
        + (1.0-y_targ)*jnp.log(1.0-y_pred)
        )

def BC_l1(y_targ, y_pred, params, lam, eps=1e-12):
    return BC_entropy(y_targ, y_pred, eps) + lam * l1_penalty(params)

def BC_l2(y_targ, y_pred,params,lam, eps=1e-12):
    return BC_entropy(y_targ,y_pred,eps) + lam * l2_penalty(params)

# -------------- 2.5 Log loss function derivatives -----------------
def dBC_dypred(y_targ, y_pred, eps=1e-12):
    y_pred = jnp.clip(y_pred, eps, 1.0 - eps)
    return (y_pred - y_targ) / (y_pred * (1.0 - y_pred) * y_targ.size)

# However the literature says that we actually do this because 
# We want to use the NN itself as the solution and add conditions
# to the cost as penalties...L = L_PDE + lam(L_IC + L_BC)
# So here I am unsure, but moving forward first then go back
# Change and then understand the math and code simultaneously

# ========================================================================
# 3 Multiclass cross entropy cost/loss function (Softmax loss func)
# ========================================================================

def softmax(z):
    """z has shape (n_samples, n_classes); softmax per row."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def multiclass_cross_entropy(y_targ, y_pred, eps=1e-12):
    """y_targ one-hot, y_pred from softmax. Mean over samples."""
    y_pred = jnp.clip(y_pred, eps, 1.0)
    return jnp.mean(-jnp.sum(y_targ * jnp.log(y_pred), axis=-1))

# -------------- 3.5 Softmax loss function derivatives -----------------
# NOTE: for softmax + multiclass cross-entropy, DO NOT use the above derivative.
# The combined derivative wrt the pre-softmax z is simply:
#     dC/dz = (y_pred - y_targ)   [up to the 1/n factor if using mean]
# This is what you set dC_dz = predict - target to in the backprop code.


