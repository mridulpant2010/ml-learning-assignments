# so linear regression we can do it in 2 ways
# 1. normal equation 
# 2. gradient descent

# normal equation theta(XT X(inv) XT y)
# normal equation is a direct method to find best fit line.
import numpy as np

# Sample data

def implement_normal_equation():
    X = np.array([[-1.0, 1.0, 2.0]]).T  # Shape (3, 1)
    Y = np.array([[7.0, 7.0, 21.0]]).T  # Shape (3, 1)
    # Add a column of ones to X to account for the intercept term (theta_0)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Shape (3, 2)
    # Calculate the least squares solution using the normal equation
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
    # Extract theta_0 and theta_1
    theta_0, theta_1 = theta_best[0, 0], theta_best[1, 0]
    print(f"theta_0: {theta_0}, theta_1: {theta_1}")

#XT in numpy how to find X transpose
# import numpy as np
# X = np.array([[-1,1,2]])
# Y = np.array([[7,7,21]])
# print(X.T)
# print(X)
# # calculate inverse of X
# # Check if X is square and invertible
# X_inv = np.linalg.inv(X)
# print(X_inv)
# #print(X.T.dot(X_inv))
# # what is the basic criteria to find inverse of a matrix
# # 1. the matrix should be a square matrix
# # 2. the determinant of the matrix should not be zero
# # 3. the matrix should be invertible
# # 4. the matrix should be non-singular
# # 5. the matrix should be diagonalizable
# # 6. the matrix should be orthogonal
# # 7. the matrix should be unitary
# # 8. the matrix should be positive definite

# # Calculate X transpose dot Y
# print(X.T.dot(Y))

#solution 2
# gradient descent
def implement_stochastic_gradient_descent():
    num_iterations = 1000
    X = np.array([-1.0, 1.0, 2.0])
    Y = np.array([7.0, 7.0, 21.0])
    theta_0 = 0
    theta_1 = 0
    alpha = 0.01
    for _ in range(num_iterations):
        for i in range(len(X)):
            error = (theta_0 + theta_1 * X[i]) - Y[i]
            theta_0 -= alpha * error
            theta_1 -= alpha * error * X[i]
    print(theta_0, theta_1)
