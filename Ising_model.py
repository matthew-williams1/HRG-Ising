from Lattice_Class import Lattice # Imports the Lattice class
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as anim
import h5py # Imports the library used for storing data

file = h5py.File("Lattice.hdf5", "w")  # Initialize a file object to save the cooling history of the lattice.

a = 0             # Relative strength of diagonal particles on a given particle
width = 40        # Initializing width of lattice
height = 40       # Initializing length of lattice
temperature = 12  # Starting temperature for simulation (in Kelvin)

# Initializing the animation plot
fig = plt.figure()
fig.clear()

# Performs one Monte Carlo step and visualizes the result while lowering the temp
def animate(i):
    fig.clear()
    lattice.monte_carlo(1)
    temperature_string = "Temperature: " + str(lattice._temperature)
    fig.suptitle(temperature_string)
    lattice.visualize(filtered=True)

# Initializing lattice variable
lattice = Lattice(width, height, temperature)

# Initializing animation variable
anim = anim.FuncAnimation(fig, animate)
plt.show()

# Visualizing magnetic field
plt.imshow(lattice.get_magnetic_field(), cmap='winter', interpolation='None')
plt.show()


plt.imshow((lattice.get_running_sum() - np.amin(lattice.get_running_sum())) /
           (np.amax(lattice.get_running_sum()) - np.amin(lattice.get_running_sum())),
           cmap='winter', interpolation='None')

print(lattice.mse())
plt.show()

# Initializes files to store the training data and the labels for the ANN
# train_data = file.create_dataset("train_data", data=np.packbits(lattice.getCoolingHistory(), axis=0))
# labels = file.create_dataset("labels", data=np.packbits(lattice.getMagneticField(), axis=0))
