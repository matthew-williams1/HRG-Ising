import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
import matplotlib.animation as anim

width = 25
height = 25
temp = 0.1  # Temperature is in Kelvin


# plt.ion()

class Lattice(object):
    '''Class to represent a lattice'''

    def __init__(self, width, height, temperature):
        '''Initializes the lattice'''
        self._width = width
        self._height = height
        self._temperature = temperature
        self._matrixRepresentation = np.random.choice([-1, 1], size=(height, width))
        self._energy = self.energyCalculation(self._matrixRepresentation)

    def energyCalculation(self, lattice):
        '''Calculates the total energy of the lattice'''
        lattice = self._matrixRepresentation
        upshift = np.roll(lattice, -1, axis=0)
        downshift = np.roll(lattice, 1, axis=0)
        leftshift = np.roll(lattice, -1, axis=1)
        rightshift = np.roll(lattice, 1, axis=1)

        return -(np.sum(lattice * (upshift + downshift + leftshift + rightshift))) / 2

    def monteCarlo(self, steps):
        '''Performs Monte Carlo algorithm'''

        for x in range(steps):
            i = random.randint(0,self._width-1)
            j = random.randint(0, self._height-1)

            energy1 = self._energy
            self._matrixRepresentation[i][j] *= -1
            energy2 = self.energyCalculation(self._matrixRepresentation)

            print(energy1 - energy2)
            if energy1 >= energy2: #and np.random.rand() < np.exp(-(energy2 - energy1) / Boltzmann / (self._temperature)):
                self._matrixRepresentation[i][j] *= -1


    def width(self):
        '''Returns the width of the lattice'''
        return self._width

    def height(self):
        '''Returns the height of the lattice'''
        return self._height

    def energy(self):
        '''Returns the energy of the lattice'''
        return self._energy

    def getMatrixRepresentation(self):
        '''Returns the matrix representation of the lattice'''
        return self._matrixRepresentation

    def visualize(self):
        '''Visualizes the the lattice as a colour map'''
        plt.imshow(self._matrixRepresentation, cmap='summer', interpolation='nearest')

    def __repr__(self):
        '''Returns a string representation of the lattice'''
        return str(self._matrixRepresentation)


def animate(i):
    '''Function called every time a frame is made in the animation. Used for FuncAnimation.'''
    fig1.clear()
    lattice.monteCarlo(1000)
    lattice.visualize()


lattice = Lattice(width, height, temp)
fig1 = plt.figure()
animation = anim.FuncAnimation(fig1, animate)

plt.show()

