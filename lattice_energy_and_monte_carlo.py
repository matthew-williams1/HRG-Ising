import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
import matplotlib.animation as anim

width = 25
height = 25
temp = 2 #Temperature is in Kelvin
#plt.ion()

class Lattice(object):
    '''Class to represent a lattice'''
    def __init__(self, width, height, temperature):
        '''Initializes the lattice'''
        self._width = width
        self._height = height
        self._temperature = temperature
        self._matrixRepresentation = np.random.randint(2, size=(height, width))
        self._energy = self.energyCalculation(self._matrixRepresentation)
        
    def energyCalculation(self, lattice):
        '''Calculates the total energy of the lattice'''
        lattice = 2*lattice - np.ones((self._height, self._width))
        upshift = np.roll(lattice, -1, axis=0)
        downshift = np.roll(lattice, 1, axis=0)
        leftshift = np.roll(lattice, -1, axis=1)
        rightshift = np.roll(lattice, 1, axis=1)

        return (np.sum(lattice*upshift) + np.sum(lattice*downshift) + np.sum(lattice*leftshift) + np.sum(lattice*rightshift))/2

    def monteCarlo(self, steps):
        '''Performs Monte Carlo algorithm'''
        for x in range(steps):
            i = random.randint(0,self._width-1)
            j = random.randint(0, self._height-1)

            latticeFlippedMatrix = np.copy(self._matrixRepresentation)
            
            if latticeFlippedMatrix[i][j] == 1:
                latticeFlippedMatrix[i][j] = 0
            else:
                latticeFlippedMatrix[i][j] = 1

            energyFlipped = self.energyCalculation(latticeFlippedMatrix)

            if energyFlipped < self._energy:
                self._matrixRepresentation = latticeFlippedMatrix
            else: #What to do if new energy is GREATER than original energy
                #This doesn't work
                rand = random.uniform(0,1)
                energyDifference = energyFlipped-self._energy
                if rand > np.exp(-energyDifference/(self._temperature)):
                    self._matrixRepresentation = latticeFlippedMatrix
        lattice.visualize()
        
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
        plt.imshow(self._matrixRepresentation, cmap = 'winter', interpolation='nearest')
        plt.show()

    def __repr__(self):
        '''Returns a string representation of the lattice'''
        return str(self._matrixRepresentation)

lattice = Lattice(width,height, temp)
#print(lattice)
#print(lattice.energy())
lattice.visualize()

lattice.monteCarlo(10000)

#fig1 = plt.figure()
#animation = anim.FuncAnimation(fig1, monteCarlo, interval=50, fargs(1,))
