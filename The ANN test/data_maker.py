import numpy as np
from LatticeClass import Lattice
import h5py

file = h5py.File("cooling_history.hdf5", "w")

# Set some parameters for the data to be fed to the network.
num_data_pts = 250  # Number of samples. 250 cooling histories. Each contains 5 images, can be changed in latticeclass.
num_test_pts = int(0.2 * num_data_pts)  # Test data, could just partition the data to not train on certain samples.
size = 40  # Size of lattice. size x size. Assuming it's a square. Started at 128x128, but then each data pt took 18s.
temp = 3  # Starting temperature for cooling histories.

if __name__ == "__main__":

    # Make empty numpy arrays which will be filled with the data. The neural network takes float32 type numbers,
    # so we will cast that now.
    dataset = np.empty((num_data_pts, 5, size, size), dtype=np.float32)
    testset = np.empty((num_test_pts, 5, size, size), dtype=np.float32)
    test_targets = np.empty((num_test_pts, 3, 3))
    targets = np.empty((num_data_pts, 3, 3), dtype=np.float32)

    # Make one data pt per iteration, each data pt has 5 images and one magnetic field (the image).
    for iteration in range(num_data_pts):
        lattice = Lattice(size, size, temp)

        while lattice.temperature() > 0.1:

            lattice.monte_carlo(5)

        dataset[iteration], targets[iteration] = lattice.get_data_set()
        print("Iteration", iteration + 1)

    # Same thing but for test data. No validation set for now.
    for iteration in range(num_test_pts):
        lattice = Lattice(size, size, temp)

        while lattice.temperature() > 0.1:
            lattice.monte_carlo(5)

        testset[iteration], test_targets[iteration] = lattice.get_data_set()
        print("Iteration", iteration + 1)

    # Separate the data into h5py datasets for easier retrieval later.
    dset = file.create_dataset("data", data=dataset)
    tset = file.create_dataset("targets", data=targets)
    testset = file.create_dataset("test_data", data=testset)
    testtargets = file.create_dataset("test_targets", data=test_targets)
