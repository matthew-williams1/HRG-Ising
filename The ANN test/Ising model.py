from LatticeClass import Lattice
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as anim
import h5py

file = h5py.File("Lattice.hdf5", "w")  # Initialize a file object to save the cooling history of the lattice.

a = 0
width = 100
height = 100
temperature = 6  # Temperature is in Kelvin

fig = plt.figure()
fig.clear()


def animate(i):
    lattice.monte_carlo(1)
    temperature_string = "Temperature: " + str(lattice._temperature)
    fig.suptitle(temperature_string)
    lattice.visualize(filtered=False)


lattice = Lattice(width, height, temperature)

anim = anim.FuncAnimation(fig, animate)

plt.show()

# train_data = file.create_dataset("train_data", data=np.packbits(lattice.getCoolingHistory(), axis=0))
# labels = file.create_dataset("labels", data=np.packbits(lattice.getMagneticField(), axis=0))
