from LatticeClass import Lattice
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as anim
import h5py

file = h5py.File("Text_write.hdf5", "w")  # Initialize a file object to save the cooling history of the lattice.

a = 0
width = 200
height = 200
temperature = 2  # Temperature is in Kelvin


def animate(i):
    '''Function called every time a frame is made in the animation. Used for FuncAnimation.'''
    fig1.clear()
    lattice.monteCarlo(3)
    temperature_string = "Temperature: " + str(lattice.temperature())
    fig1.suptitle(temperature_string)
    lattice.visualize()


lattice = Lattice(width, height, temperature)
fig1 = plt.figure()
animation = anim.FuncAnimation(fig1, animate)

plt.show()

train_data = file.create_dataset("train_data", data=np.packbits(lattice.getCoolingHistory(), axis=0))
labels = file.create_dataset("labels", data=np.packbits(lattice.getMagneticField(), axis=0))
