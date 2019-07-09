import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
import matplotlib.animation as anim

width = 20
height = 20
temperature = 2  # Temperature is in Kelvin
cooling_history = open(r"Cooling_History.txt", "a") # opens .txt file to store lattice configs

class Lattice(object):
    '''Class to represent a lattice'''

    def __init__(self, width, height, temperature):
        '''Initializes the lattice'''
        self._width = width
        self._height = height
        self._temperature = temperature
        self._matrixRepresentation = np.rint(np.random.choice([-1, 1], size=(height, width)))
        self._energy = self.energyCalculation(self._matrixRepresentation)

    def make_field(self):
        '''Returns a field matrix with two uniform halves'''
        half1 = random.uniform(-1, 1)
        half2 = random.uniform(-1, 1)
        field = np.empty((width, height))

        for i in range(height):
            for j in range(int(width/2)):
                field[i][j] = half1

            for j in range(int(width/2), width):
                field[i][j] = half2

        return field

    def energyCalculation(self, lattice):
        '''Calculates the total energy of the lattice'''
        # Shifting the lattice in order to get the nearest
        # neighbour interactions as efficiently as possible
        lattice = self._matrixRepresentation
        upshift = np.roll(lattice, -1, axis=0)
        downshift = np.roll(lattice, 1, axis=0)
        leftshift = np.roll(lattice, -1, axis=1)
        rightshift = np.roll(lattice, 1, axis=1)
        # Magnitude of the external electric field
        # H = 100*np.random.rand(self._height, self._width)
        # print("H=",H)
        # print("Field energy=",np.sum(H*lattice))

        return -(np.sum(lattice * (upshift + downshift + leftshift + rightshift))) / 2 - (np.sum(self.make_field()*lattice))

    def energy_at_a_point(self, i, j):
        '''Calculates the energy at a given point'''
        return -self._matrixRepresentation[i][j] * (self._matrixRepresentation[(i-1) % self._width][j % self._width]
                                                   + self._matrixRepresentation[(i+1) % self._width][j % self._width]
                                                   + self._matrixRepresentation[i % self._width][(j-1) % self._width]
                                                   + self._matrixRepresentation[i % self._width][(j+1) % self._width])

    def monteCarlo(self, steps):
        '''Performs Monte Carlo algorithm'''
        for x in range(steps):
            #self._energy = self.energyCalculation(self._matrixRepresentation)

            i = random.randint(0,self._width-1)
            j = random.randint(0, self._height-1)
            r = random.uniform(0,1)

            energy1 = self.energy_at_a_point(i,j)
            self._matrixRepresentation[i][j] *= -1
            energy2 = self.energy_at_a_point(i,j)

            prob = self.probability(energy1-energy2)

            transitionProbability = min(1, prob)
            if r > transitionProbability:
                self._matrixRepresentation[i][j] *= -1

            if self._temperature > 0.1:
                self._temperature-=0.001

        # saves the lattice config to an uncompressed .txt file
        np.savetxt(cooling_history, self._matrixRepresentation, fmt = '%.01e', newline='\n')
        # saves the lattice config to a compressed .npz file
        np.savez_compressed('Compressed_Cooling_History', self._matrixRepresentation)

    def probability(self, energy):
        '''Calculates the probability given an energy'''
        return np.exp(-energy/self._temperature)

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

    def temperature(self):
        '''returns the temperature'''
        return self._temperature


def animate(i):
    '''Function called every time a frame is made in the animation. Used for FuncAnimation.'''
    fig1.clear()
    lattice.monteCarlo(1000)
    temperature_string = "Temperature: " + str(lattice._temperature)
    fig1.suptitle(temperature_string)
    lattice.visualize()

lattice = Lattice(width, height, temperature)
fig1 = plt.figure()
animation = anim.FuncAnimation(fig1, animate)

plt.show()

#print(np.load('arr_0.npy'))

cooling_history.close()
