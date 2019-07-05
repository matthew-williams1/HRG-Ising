import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import sys
from matplotlib import style

style.use('dark_background')
num_range = 6
num_points = 30
num_terms = 10

fig, ax = plt.subplots()

ax2 = fig.add_subplot(1,1,1)


def make_values(num_terms, num_points):
    values = np.zeros(num_points)
    for i in range(num_points):
        values[i] = num_range * i / num_points

    return values


def make_X(num_terms, num_points):
    X = np.zeros((num_points, num_terms))
    for i in range(num_points):
        for j in range(1, num_terms+1):
            if j % 2 == 0:
                X[i][j-1] = math.sin(j * num_range * i / (2 * num_points))
            else:
                X[i][j-1] = math.cos((j-1) * num_range * i / (2 * num_points))

    return X


def make_y(num_terms, num_points):
    Y = np.zeros(num_points)
    for i in range(num_points):
        Y[i] = (num_range / num_points) * i #+ np.random.normal(0, 0.1)

    return Y


def get_grad(alpha, X, Y):
    gradient = np.zeros((num_terms, 1))

    for k in range(num_terms):
        sum = 0
        for i in range(num_points):
            sum += (Y[i] - (X[i] @ alpha)) * X[i][k]
        gradient[k] = 2 * sum

    return gradient


def animate(i):
    optimize(X, y, alpha, 100)
    ax2.clear()
    ax2.plot(x_val, y, 'co')
    ax2.plot(x_val, X.dot(alpha), 'm:')
    if i > 20:
        sys.exit()
    return curve


def optimize(x_set, y_set, alpha, epochs, lr=0.0001):
    for epoch in range(epochs):
        alpha += lr * get_grad(alpha, x_set, y_set)


X = make_X(num_terms, num_points)
y = make_y(num_terms, num_points)
x_val = make_values(num_terms, num_points)
alpha = np.random.rand(num_terms, 1)

line, = ax.plot(x_val, x_val)
curve, = ax2.plot(x_val, X @ alpha)


ani = animation.FuncAnimation(fig, animate)

plt.show()


