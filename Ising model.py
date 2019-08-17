from LatticeClass import Lattice
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as anim
import h5py

file = h5py.File("Lattice.hdf5", "w")  # Initialize a file object to save the cooling history of the lattice.

a = 0
width = 40
height = 40
temperature = 12  # Temperature is in Kelvin

fig = plt.figure()
fig.clear()


def animate(i):
    fig.clear()
    lattice.monte_carlo(1)
    temperature_string = "Temperature: " + str(lattice._temperature)
    fig.suptitle(temperature_string)
    lattice.visualize(filtered=True)


lattice = Lattice(width, height, temperature)

anim = anim.FuncAnimation(fig, animate)
plt.show()

plt.imshow(lattice.get_magnetic_field(), cmap='winter', interpolation='None')
plt.show()

plt.imshow((lattice.get_running_sum() - np.amin(lattice.get_running_sum())) /
           (np.amax(lattice.get_running_sum()) - np.amin(lattice.get_running_sum())),
           cmap='winter', interpolation='None')
plt.show()


# train_data = file.create_dataset("train_data", data=np.packbits(lattice.getCoolingHistory(), axis=0))
# labels = file.create_dataset("labels", data=np.packbits(lattice.getMagneticField(), axis=0))
