import random
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal

a = 0


class Lattice(object):
    '''Class to represent a lattice'''

    def __init__(self, width, height, temperature):
        '''Initializes the lattice'''
        self._width = width
        self._height = height
        self._temperature = temperature
        self._savePeriod = 1
        self._matrix_representation = np.rint(np.random.choice([-1, 1], size=(height, width)))
        self._field, self._answer = self.make_field()
        self._history = np.zeros((5, width, height))
        self._energy = self.energy_calculation(self._matrix_representation)
        self._slide = 0

    def make_field(self):
        '''Returns a field matrix with two uniform halves'''
        occupied = np.random.randn(3, 3)  # Define 3x3 array with random values from a normal dist. for the
        # magnetic field. Serves also as the answer for the neural network.

        field = np.ones((self._width, self._height))  # actual field which will be filled with numbers.

        block_width = int(self._width / 3) - 1

        # Fill the field with values. If the square from the 3x3 has a value >0.3, then it will be active and the value
        # will be increased.
        for i in range(occupied.shape[0]):
            for j in range(occupied.shape[1]):
                if occupied[i, j] > 0.3:
                    occupied[i, j] *= np.random.uniform(2, 6)
                    for row in range(i*block_width, (i+1)*block_width + 1):
                        for col in range(j*block_width, (j+1)*block_width+1):
                            field[row, col] = occupied[i, j]

        return field, occupied

    def energy_calculation(self, lattice):
        '''Calculates the total energy of the lattice'''
        # Shifting the lattice in order to get the nearest
        # neighbour interactions as efficiently as possible
        lattice = self._matrix_representation
        upshift = np.roll(lattice, -1, axis=0)
        downshift = np.roll(lattice, 1, axis=0)
        leftshift = np.roll(lattice, -1, axis=1)
        rightshift = np.roll(lattice, 1, axis=1)
        # Magnitude of the external magnetic field
        # print("H=",H)
        # print("Field energy=", np.sum(H*lattice))

        return -(np.sum(lattice * (upshift + downshift + leftshift + rightshift))) / 2 - (np.sum(self._field*lattice))

    def energy_at_a_point(self, i, j):
        '''Calculates the energy at a given point'''
        return -(self._matrix_representation[i][j] * (self._matrix_representation[(i-1) % self._width][j % self._width]
                                            + self._matrix_representation[(i+1) % self._width][j % self._width]
                                            + self._matrix_representation[i % self._width][(j-1) % self._width]
                                            + self._matrix_representation[i % self._width][(j+1) % self._width]
                                            + a * (self._matrix_representation[(i-1) % self._width][(j-1) % self._width]
                                            + self._matrix_representation[(i+1) % self._width][(j-1) % self._width]
                                            + self._matrix_representation[(i+1) % self._width][(j+1) % self._width]
                                            + self._matrix_representation[(i-1) % self._width][(j+1) % self._width])) / 2
                                            - self._field[i][j])

    def monte_carlo(self, steps):
        '''Performs Monte Carlo algorithm'''
        for x in range(steps):
            for i in range(self._width):
                for j in range(self._height):
                    r = random.uniform(0,1)
                    energy1 = self.energy_at_a_point(i,j)
                    self._matrix_representation[i][j] *= -1
                    energy2 = self.energy_at_a_point(i,j)

                    prob = self.probability((energy1-energy2) * self._field[i , j])

                    if r > min(1, prob):
                        self._matrix_representation[i][j] *= -1

            if self._temperature > 0.1:
                self._temperature -= 0.05
            if (2 * self._temperature - int(2 * self._temperature)) < 0.0601:
                if self._slide < 5:
                    self._history[self._slide] = self._matrix_representation
                self._slide += 1

    def probability(self, energy):
        '''Calculates the probability given an energy'''
        return np.exp(-energy/self._temperature)

    def sobel_filter(self, source_image):
        '''Convolves the lattice'''
        kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        return scipy.signal.convolve2d(kernel, source_image, boundary='fill')

    def width(self):
        '''Returns the width of the lattice'''
        return self._width

    def height(self):
        '''Returns the height of the lattice'''
        return self._height

    def energy(self):
        '''Returns the energy of the lattice'''
        return self._energy

    def get_matrix_representation(self):
        '''Returns the matrix representation of the lattice'''
        return self._matrix_representation

    def get_cooling_history(self):
        '''Returns the cooling history array of the lattice.'''
        return self._history

    def get_magnetic_field(self):
        '''Returns the magnetic field of the lattice.'''
        return self._field

    def visualize(self):
        '''Visualizes the the lattice as a colour map'''
        plt.imshow(self._matrix_representation, cmap='winter', interpolation='nearest')

    def visualize_filtered(self):
        '''Visualizes the the lattice as a colour map'''
        plt.imshow(self.sobel_filter(self._matrix_representation), cmap='winter', interpolation='nearest')

    def __repr__(self):
        '''Returns a string representation of the lattice'''
        return str(self._matrix_representation)

    def temperature(self):
        '''returns the temperature'''
        return self._temperature

    def get_data_set(self):
        return self._history, self._answer
