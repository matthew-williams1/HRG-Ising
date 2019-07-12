import random
import numpy as np
from matplotlib import pyplot as plt

corner_factor = 0
width = 200
height = 200
temperature = 2  # Temperature is in Kelvin


class Lattice(object):
    '''Class to represent a lattice'''

    def __init__(self, width, height, temperature):
        '''Initializes the lattice'''
        self._width = width
        self._height = height
        self._temperature = temperature
        self._savePeriod = 1
        self._matrixRepresentation = np.rint(np.random.choice([-1, 1], size=(height, width)))
        self._field = self.make_field()
        self._history = np.zeros(())
        self._energy = self.energyCalculation(self._matrixRepresentation)

    def make_field(self):
        '''Returns a field matrix with two uniform halves'''
        half1 = -5
        half2 = 5
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
        # Magnitude of the external magnetic field
        H = 100*np.random.rand(self._height, self._width)
        print("H=",H)
        print("Field energy=", np.sum(H*lattice))

        return -(np.sum(lattice * (upshift + downshift + leftshift + rightshift))) / 2 - (np.sum(self._field*lattice))

    def energy_at_a_point(self, i, j):
        '''Calculates the energy at a given point'''
        return -(self._matrixRepresentation[i][j] * (self._matrixRepresentation[(i-1) % self._width][j % self._width]
                                            + self._matrixRepresentation[(i+1) % self._width][j % self._width]
                                            + self._matrixRepresentation[i % self._width][(j-1) % self._width]
                                            + self._matrixRepresentation[i % self._width][(j+1) % self._width]
                                            + corner_factor * (self._matrixRepresentation[(i-1) % self._width][(j-1) % self._width]
                                            + self._matrixRepresentation[(i+1) % self._width][(j-1) % self._width]
                                            + self._matrixRepresentation[(i+1) % self._width][(j+1) % self._width]
                                            + self._matrixRepresentation[(i-1) % self._width][(j+1) % self._width])) / 2
                                            - self._field[i][j])

    def monteCarlo(self, steps):
        '''Performs Monte Carlo algorithm'''
        for x in range(steps):
            for i in range(width):
                for j in range(height):
                    r = random.uniform(0,1)
                    energy1 = self.energy_at_a_point(i,j)
                    self._matrixRepresentation[i][j] *= -1
                    energy2 = self.energy_at_a_point(i,j)

                    prob = self.probability(energy1-energy2)

                    transitionProbability = min(1, prob)
                    if r > transitionProbability:
                        self._matrixRepresentation[i][j] *= -1

            if self._temperature > 0.1:
                self._temperature -= 0.01
                
    def sobel_filter(self, source_image):
        '''Convolves the lattice'''
        kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        return scipy.signal.convolve2d(kernel, source_image, boundary='fill')

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

    def getCoolingHistory(self):
        '''Returns the cooling history array of the lattice.'''
        return self._history

    def getMagneticField(self):
        '''Returns the magnetic field of the lattice.'''
        return self._field

    def visualize(self):
        '''Visualizes the the lattice as a colour map'''
        plt.imshow(self._matrixRepresentation, cmap='summer', interpolation='nearest')

    def __repr__(self):
        '''Returns a string representation of the lattice'''
        return str(self._matrixRepresentation)

    def temperature(self):
        '''returns the temperature'''
        return self._temperature
