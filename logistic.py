import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''

def logistic(w,xTr,yTr):

    # YOUR CODE HERE
    y_pred = np.dot(w.T, xTr)

    loss = np.sum(np.log(1 + np.exp(-1 * yTr * y_pred)))


    gradient = -1 * np.dot(xTr, (yTr / (1 + np.exp(yTr * y_pred))).T)
    return loss,gradient
