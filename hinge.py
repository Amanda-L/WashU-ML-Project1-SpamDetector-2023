from numpy import maximum
import numpy as np


#
#
# INPUT:
# xTr dxn matrix (each column is an input vector) 1024 * 4000
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w

def hinge(w,xTr,yTr,lambdaa):
    gradient = np.zeros_like(w)
    loss = 0
    reg = lambdaa * np.sum(w * w)

    # xTr = (1024, 4000), w = (1024, 1), ytr = (1, 4000)

    dotProduct = np.dot(w.T, xTr)  #this is (1, 4000)


    margin = 1 - yTr*dotProduct # (1,4000) array
    
    loss = np.sum(np.maximum(0, margin)) + reg
    
    y_pred_flag = (yTr * np.dot(w.T, xTr)) < 1
    
    gradient = -1 * np.dot(xTr, (yTr * (y_pred_flag)).T) + 2 * lambdaa * w
    
        
    return loss, gradient

    


