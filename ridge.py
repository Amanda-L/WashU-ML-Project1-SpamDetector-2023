
import numpy as np

#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);
def ridge(w,xTr,yTr,lambdaa):
    n = xTr.shape[1]
    reg = lambdaa * np.dot(w.T, w)

    loss = np.sum((yTr - np.dot(w.T, xTr)) ** 2) + reg

    gradient = -2 * np.dot(xTr, (yTr - np.dot(w.T, xTr)).T) + 2 * lambdaa * w
    

    return loss,gradient
