# Project 1: Structural Risk Minimization

![image](https://github.com/Amanda-L/WashU-ML-Project1-StructuralRiskMinimization-2023/assets/52643725/f01ddf24-fdce-4bfc-bf3c-71d4fc6875a2)


The assignment was to build an email spam detection, tasked with implementing and testing various functions related to gradient descent and classification algorithms. Here is a summarized breakdown of the tasks:

1. **Data Loading and Splitting:**
   - Load the training data from 'data_train.mat' using `io.loadmat` in `project1Main.py`.
   - Split the data into training (`xTr`, `yTr`) and validation (`xTv`, `yTv`) sets with 4000 and 1000 data points, respectively.

2. **Ridge Regression:**
   - Implement the `ridge.py` function to compute the loss and gradient for a dataset using ridge regression.
   - Incorporate a regularization constant in the computation.
   - Check the gradient using the code in `checkgradHingeAndRidge.py`.

3. **Gradient Descent:**
   - Implement the `grdescent.py` function for gradient descent.
   - Include a tolerance variable to stop early if the norm of the gradient is below a specified value.
   - Choose a step-size strategy for gradient descent.

4. **Linear Model Prediction:**
   - Write the `linearmodel` function to return predictions for a weight vector `w` and a dataset `xTv`.

5. **Main Module Execution:**
   - Call the main module using `python3 project1Main.py`.
   - Train a spam filter and save the resulting weight vector in `w_trained.mat`.
   - Evaluate the spam filter on the validation dataset.

6. **Objective Function and Parameters:**
   - Find a suitable objective function and input parameters for both the objective function and the gradient descent method.

7. **False Positive Rate, True Positive Rate, and AUC:**
   - Understand and interpret the outputs of the spam filter, including false positive rate, true positive rate, and area under the curve (AUC).

8. **Demo and Misclassified Emails:**
   - Use `spamdemo.py` to identify misclassified emails and observe which emails get classified incorrectly.

9. **Implement Hinge and Logistic Loss:**
   - Implement the `hinge.py` function, equivalent to ridge regression but with hinge loss.
   - Implement the `logistic.py` function, equivalent to ridge regression but with log-loss (logistic regression).
   - Check the gradients using provided code in `checkgradHingeAndRidge.py` and `checkgradLogistic.py`.

10. **Visualization:**
    - Run `vis_rocs` to visualize the performance of the implemented algorithms.
    - Adjust the `STEPSIZE` parameter as needed.

11. **Final Spam Filter:**
    - Modify `trainspamfilter.py` to use the desired loss function, settings, and parameters for the final spam filter.
    - Train the spam filter by running `project1Main.py` or `spamdemo.py`.


View 01SRM.html in the instructions folder for detailed instructions on the assignment.
