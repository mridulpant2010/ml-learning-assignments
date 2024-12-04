# so linear regression we can do it in 2 ways
# 1. normal equation 
# 2. gradient descent

# normal equation theta(XT X(inv) XT y)
# normal equation is a direct method to find best fit line.
import numpy as np

# Sample data
def implement_normal_equation():
    X = np.array([[1, 0],
                  [-1, 1],
                  [0, 1],
                  ])  # Shape (5, 2)
    Y = np.array([1.0, 0.0, 1,0])  # Shape (5, 1)
    # Add a column of ones to X to account for the intercept term (theta_0)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Shape (5, 3)
    print(X_b)
    # Calculate the least squares solution using the normal equation
    try:
        inve = X_b.T.dot(X_b)
        second = X_b.T.dot(Y)
        print(inve)
        print(second)
        theta_best = np.linalg.inv(inve).dot(second)
    except np.linalg.LinAlgError:
        print("Error: Singular matrix encountered. Please check the data.")
        return
    # Extract theta_0, theta_1, and theta_2
    theta_0, theta_1, = theta_best
    print(f"theta_0: {theta_0:.4f}, theta_1: {theta_1:.4f}, theta_2: {theta_2:.4f}")

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
        sum_errors_0 = 0
        sum_errors_1 = 0
        for i in range(len(X)):
            error = (theta_0 + theta_1 * X[i]) - Y[i]
            sum_errors_0 += error
            sum_errors_1 += error * X[i]
        theta_0 -= alpha * (1 / len(X)) * sum_errors_0
        theta_1 -= alpha * (1 / len(X)) * sum_errors_1
    print(theta_0, theta_1)



import numpy as np

def multiple_regression(X, y):
    """
    Performs multiple regression using matrix form.

    Args:
        X: A numpy array representing the independent variables (features).
        y: A numpy array representing the dependent variable (target).

    Returns:
        beta: A numpy array representing the regression coefficients.
    """

    # Check if X is singular
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("X is singular and cannot be inverted.")

    # Calculate the coefficients using the matrix formula
    try:
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
    except np.linalg.LinAlgError as e:
        raise ValueError("X is singular and cannot be inverted.") from e

    return beta


if __name__ == "__main__":
    implement_normal_equation()
    # Example usage:
    # X = np.array([[1, 2],
    #           [2, 4],
    #           [3, 6],
    #           [4, 8],
    #           [5, 10]])
    # #X = np.array([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
    # y = np.array([5, 7, 10, 12, 15])

    # coefficients = multiple_regression(X, y)
    # print(coefficients)
