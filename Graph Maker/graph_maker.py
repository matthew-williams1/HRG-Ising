import matplotlib.pyplot as plt
import numpy as np

mse = []
strengths = []

f = open("mse.txt", "r")
for line in f:
    mse.append(float(line.strip('\n')))
f.close()

f = open("strengths.txt", "r")
for line in f:
    strengths.append(float(line.strip('\n')))
f.close()

coefficients = np.polyfit(strengths, mse, 3)
x = []
y = []

f = np.poly1d(coefficients)

x_new = np.linspace(0.7, 10.25, 50)
y_new = f(x_new)

axes = plt.gca()
axes.set_xlim([0.5, 10.25])
axes.set_ylim([0.06, 0.375])

plt.plot(x_new, y_new, color='orange', lw=2.3)
plt.scatter(strengths, mse, s=0.5)
plt.title('MSE vs Magnitude of Magnetic Field after 240 Monte Carlo Steps')
plt.xlabel('Magnetic Field Strength (T)')
plt.ylabel('Mean Squared Error')
plt.show()
