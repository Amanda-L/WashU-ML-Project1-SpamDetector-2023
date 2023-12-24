
import numpy as np
# import matplotlib.pyplot as plt
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent
    
    loss, gradient = func(w0)
    w = w0

    for i in range(maxiter):

        loss0 = loss
        # Check if the norm of the gradient is less than the tolerance
        if np.linalg.norm(gradient) < tolerance:
            return w        
        # Update the weight vector
        w = w - stepsize * gradient    
        
        loss, gradient = func(w)

        if loss < loss0:
            stepsize *= 1.01            
        elif loss > loss0:
            stepsize *= 0.50
            if stepsize < eps:
                print("stepsize < eps")
                break

    return w
