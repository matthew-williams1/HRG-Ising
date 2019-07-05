import numpy as np
import random
import math

dimensions = 10
num_samples = 25


true_coeff = np.random.normal(0, 2, dimensions)
intercept = random.uniform(-5, 5)


def f(x):
    return true_coeff @ np.transpose(x) + intercept


def make_test_data(m, samples):
    data_x = np.zeros((samples, m))
    data_y = np.zeros((samples))
    for i in range(samples):
        for j in range(m):
            data_x[i][j] = random.randint(0, 100)
        data_y[i] = f(data_x[i]) + np.random.normal(0, 5)  #Add a bit of noise to outputs to not get exact values of coefficients.

    return data_x, data_y


def LinRegressionSolver(inputArray, outputArray):

    intercept_column = np.ones((num_samples, 1))

    inputArray = np.append(inputArray, intercept_column, axis=1)

    return np.linalg.inv(np.transpose(inputArray) @ inputArray) @ np.transpose(inputArray) @ outputArray


dataset_x, dataset_y = make_test_data(dimensions, num_samples)


print("True coefficients: ", true_coeff)
print("Intercept: ", intercept)

answers = LinRegressionSolver(dataset_x, dataset_y)
print("Solved coefficients: ", answers[:-1])
print("Solved intercept: ", answers[-1])

true_coeff = np.append(true_coeff, intercept)
print("RMSE of coefficients = ", math.sqrt(((true_coeff - answers)**2).sum() / (dimensions + 1)))