# HRG-Ising

This repository is for the code written by the Hernandez Research Group.
Our research project is to create a machine learning algorithm that can
predict the magnetic field acting on an Ising model based solely on the
cooling history of this Ising model.

We now have a functional Ising model simulation using the Gibbs sampling 
method. Our model simulates an anti-ferromagnetic lattice, so it favours 
a "checkerboard" configuration. We have also added the necessary code 
to simulate the external magnetic field.

The next step is to decide which variables we want to store as the cooling 
history of the model in order to train the neural network, produce the 
training sets, and build the neural network that will predict the magnitude
of the external magnetic field.
