import random
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal
import pandas as pd

a = 0


class Lattice(object):
    """Class to represent a lattice"""

    def __init__(self, width, height, temperature):
        """Initializes the lattice"""
        self._width = width
        self._height = height
        self._temperature = temperature
        self._save_period = 1
        self._matrix_representation = np.rint(np.random.choice([-1, 1], size=(height, width)))
        self._field = self.make_field()
        self._history = np.zeros((5, width, height))
        self._energy = self.energy_calculation()

        self._running_sum = scipy.signal.convolve2d(self._matrix_representation, np.array([[0,1,0],[1,0,1],[0,1,0]]), mode='same', boundary='fill') * self._matrix_representation

    def make_field(self):
        """Returns a field matrix with two uniform halves"""
        mag_field = pd.read_csv("/Users/nicholasd./Desktop/git/wip/HRG-Ising/The ANN Test/data/train/labels.csv",
                                usecols=list(np.arange(0, 40)), header=None).to_numpy().reshape(1000, 40, 40)
        '''
        occupied = np.array([[3,0,2.5],[4,0,-4],[0,0,1]])#np.random.randn(3, 3)  # Define 3x3 array with random values from a normal dist. for the
        # magnetic field. Serves also as the answer for the neural network.

        field = np.ones((self._width, self._height))  # actual field which will be filled with numbers.

        block_width = int(self._width / 3) - 1

        # Fill the field with values. If the square from the 3x3 has a value >0.3, then it will be active and the value
        # will be increased.
        for i in range(occupied.shape[0]):
            for j in range(occupied.shape[1]):
                if occupied[i, j] > 0.3:
                    occupied[i, j] *= 3
                    for row in range(i*block_width, (i+1)*block_width):
                        for col in range(j*block_width, (j+1)*block_width):
                            field[row, col] = occupied[i, j]

        print(field)'''
        index = np.random.randint(0, 1000)
        '''while np.amax(mag_field[index]) >= 3:
            index += 1'''
        field = mag_field[index]
        return field

    def energy_calculation(self):
        """Calculates the total energy of the lattice"""
        # Shifting the lattice in order to get the nearest
        # neighbour interactions as efficiently as possible
        lattice = self._matrix_representation
        upshift = np.roll(lattice, -1, axis=0)
        downshift = np.roll(lattice, 1, axis=0)
        leftshift = np.roll(lattice, -1, axis=1)
        rightshift = np.roll(lattice, 1, axis=1)

        return -(np.sum(lattice * (upshift + downshift + leftshift + rightshift))) / 2 - (np.sum(self._field*lattice))

    def energy_at_a_point(self, i, j):
        """Calculates the energy at a given point"""
        indices_x = np.array([(i-1) % self._width, (i+1) % self._width, i % self._width, i % self._width])

        indices_y = np.array([j % self._width, j % self._width, (j-1) % self._width, (j+1) % self._width])

        return -(2 * self._matrix_representation[i][j] * (np.sum(self._matrix_representation[indices_x, indices_y]) / 2
                                                      - self._field[i][j]))

    def monte_carlo(self, steps):
        """Performs Monte Carlo algorithm"""
        for x in range(steps):
            for i in range(self._width):
                for j in range(self._height):
                    energy_delta = self.energy_at_a_point(i, j)

                    if energy_delta <= 0:
                        self._matrix_representation[i][j] *= -1
                    elif np.random.uniform(0, 1) < self.probability(energy_delta):
                        self._matrix_representation[i, j] *= -1

            if self._temperature > 0.1:
                self._temperature -= 0.05
        self._energy = self.energy_calculation()

    def probability(self, energy):
        """Calculates the probability given an energy"""
        return 1 / (1 + np.exp(energy/self._temperature))

    def sobel_filter(self, source_image):
        """Convolves the lattice"""
        kernel = np.array([[0,1,0],[1,0,1],[0,1,0]])
        self._running_sum += scipy.signal.convolve2d(source_image, kernel, mode='same', boundary='fill') * self._matrix_representation
        return self._running_sum

    def organization(self):
        max_energy = 2 * self._width * self._height - np.sum(self._field*self._matrix_representation)
        min_energy = -2 * self._width * self._height - np.sum(self._field*self._matrix_representation)
        return (self._energy - min_energy) / (max_energy - min_energy)

    def width(self):
        """Returns the width of the lattice"""
        return self._width

    def height(self):
        """Returns the height of the lattice"""
        return self._height

    def energy(self):
        """Returns the energy of the lattice"""
        return self._energy

    def get_matrix_representation(self):
        """Returns the matrix representation of the lattice"""
        return self._matrix_representation

    def get_cooling_history(self):
        """Returns the cooling history array of the lattice."""
        return self._history

    def get_magnetic_field(self):
        """Returns the magnetic field of the lattice."""
        return self._field

    def get_running_sum(self):
        """Returns the running sum of the lattice"""
        return self._running_sum

    def visualize(self, filtered=False):
        """Visualizes the the lattice as a colour map"""
        if not filtered:
            plt.imshow(self._matrix_representation, cmap='winter', interpolation='nearest')
        else:
            plt.imshow(self.sobel_filter(self._matrix_representation), cmap='winter', interpolation='None')

    def __repr__(self):
        """Returns a string representation of the lattice"""
        return str(self._matrix_representation)

    def temperature(self):
        """returns the temperature"""
        return self._temperature

    def get_data_set(self):
        return self._history, self._answer
