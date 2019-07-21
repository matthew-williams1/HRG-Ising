import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
import scipy.signal
import matplotlib.animation as anim
from random import randint

corner_factor=0
width = 128
height = 128
temperature = 2  # Temperature is in Kelvin
cooling_history = open(r"Cooling_History.txt", "a") # opens .txt file to store lattice configs
is_filtered = False
field_magnitude = 5

class Lattice(object):
    '''Class to represent a lattice'''

    def __init__(self, width, height, temperature):
        '''Initializes the lattice'''
        self._width = width
        self._height = height
        self._temperature = temperature
        self._matrix_representation = np.rint(np.random.choice([-1, 1], size=(height, width)))
        self._energy = self.energyCalculation(self._matrix_representation)

    def make_field(self):
        '''Returns a magnetic field with random size at a random position'''
        fieldHeight = np.random.randint(1, height)
        fieldWidth = np.random.randint(1, width)
        field1 = field_magnitude*np.ones((fieldHeight, fieldWidth))
        indexHeight = np.random.randint(0, height-fieldHeight)
        indexWidth = np.random.randint(0, width-fieldWidth)
        zerosTop = np.zeros((indexHeight, width))
        zerosBottom = np.zeros((height-fieldHeight-indexHeight, width)) 
        zerosLeft = np.zeros((fieldHeight, indexWidth))
        zerosRight = np.zeros((fieldHeight, width-indexWidth-fieldWidth))
        field2 = np.concatenate((zerosLeft, field1, zerosRight), axis = 1)
        field = np.concatenate((zerosTop, field2, zerosBottom), axis = 0)

        '''Returns square in the upper left corner of lattice
        field1=np.zeros((height, width//2))
        field2 = np.ones((height//2, width//2))
        field3 = np.zeros((height//2, width//2))
        field4 = np.concatenate((field2, field3), axis =0)
        field = field_magnitude*np.concatenate((field4, field1), axis = 1)'''
        return field

    def energyCalculation(self, lattice):
        '''Calculates the total energy of the lattice'''
        # Shifting the lattice in order to get the nearest
        # neighbour interactions as efficiently as possible
        lattice = self._matrix_representation
        upshift = np.roll(lattice, -1, axis=0)
        downshift = np.roll(lattice, 1, axis=0)
        leftshift = np.roll(lattice, -1, axis=1)
        rightshift = np.roll(lattice, 1, axis=1)
        # Magnitude of the external magnetic field
        # H = 100*np.random.rand(self._height, self._width)
        # print("H=",H)
        # print("Field energy=",np.sum(H*lattice))

        return -(np.sum(lattice * (upshift + downshift + leftshift + rightshift))) / 2 #- (np.sum(self.make_field()*lattice))

    def energy_at_a_point(self, i, j):
        '''Calculates the energy at a given point'''
        return -self._matrix_representation[i][j] * (self._matrix_representation[(i-1) % self._width][j % self._width]
                                                + self._matrix_representation[(i+1) % self._width][j % self._width]
                                                + self._matrix_representation[i % self._width][(j-1) % self._width]
                                                + self._matrix_representation[i % self._width][(j+1) % self._width]
                                                + corner_factor * (self._matrix_representation[(i-1) % self._width][(j-1) % self._width]
                                                + self._matrix_representation[(i+1) % self._width][(j-1) % self._width]
                                                + self._matrix_representation[(i+1) % self._width][(j+1) % self._width]
                                                + self._matrix_representation[(i-1) % self._width][(j+1) % self._width])) - magneticField[i][j]*self._matrix_representation[i][j] #-self.make_field()[i][j]*self._matrix_representation[i][j]

    def monteCarlo(self, steps):
        '''Performs Monte Carlo algorithm'''
        for x in range(steps):
            #self._energy = self.energyCalculation(self._matrix_representation)
            #i = random.randint(0,self._width-1)
            #j = random.randint(0, self._height-1)
            for i in range(width):
                for j in range(height):
                    r = random.uniform(0,1)
                    energy1 = self.energy_at_a_point(i,j)
                    self._matrix_representation[i][j] *= -1
                    energy2 = self.energy_at_a_point(i,j)

                    prob = self.probability(energy1-energy2)

                    transitionProbability = min(1, prob)
                    if r > transitionProbability:
                        self._matrix_representation[i][j] *= -1

            if self._temperature > 0.1:
                self._temperature-=0.001

        # saves the lattice config to an uncompressed .txt file
        np.savetxt(cooling_history, self._matrix_representation, fmt = '%.01e', newline='\n')
        # saves the lattice config to a compressed .npz file
        np.savez_compressed('Compressed_Cooling_History', self._matrix_representation)

    def sobel_filter(self, source_image):
        '''Convolves the lattice'''
        kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        return scipy.signal.convolve2d(kernel_x, source_image, boundary='fill')   

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
        return self._matrix_representation

    def visualize(self):
        '''Visualizes the the lattice as a colour map'''
        plt.imshow(self._matrix_representation, cmap='summer', interpolation='nearest')

    def visualize_filtered(self):
        '''Visualizes the the lattice as a colour map'''
        plt.imshow(self.sobel_filter(self._matrix_representation), cmap='summer', interpolation='nearest')

    def __repr__(self):
        '''Returns a string representation of the lattice'''
        return str(self._matrix_representation)

    def temperature(self):
        '''returns the temperature'''
        return self._temperature


lattice = Lattice(width, height, temperature)
fig= plt.figure()
fig.clear()
def animate(i):
    lattice.monteCarlo(1)
    temperature_string = "Temperature: " + str(lattice._temperature)
    fig.suptitle(temperature_string)
    if is_filtered:
        return lattice.visualize_filtered()
    else:
        return lattice.visualize()

magneticField = lattice.make_field()
a=anim.FuncAnimation(fig, animate)



plt.show()

#print(np.load('arr_0.npy'))

cooling_history.close()

#print(lattice.make_field())
