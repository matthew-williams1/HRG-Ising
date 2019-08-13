import numpy as np
import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import h5py


class ConvNet(nn.Module):

    # Image starts at 128x128 for this test.
    def __init__(self, input_channels=15, output_dim=1, channels1=30, channels2=60, h1=200, h2=128, h3=64,
                 input_shape=40):
        super(ConvNet, self).__init__()
        self._final = channels2

        self.conv = nn.Conv2d(input_channels, output_dim, kernel_size=5, stride=1)

        self._size1 = self.convolution_dimensions(input_shape, kernel_size=5)
        self._size2 = self.convolution_dimensions(self._size1, kernel_size=5)

        self.sigmoid = nn.Sigmoid()

    def convolution_dimensions(self, h_in, kernel_size, pool_size=2, stride=1, padding=0, dilation=1):
        return int(((h_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride + 1) / pool_size)

    def forward(self, x):


    def save(self, filename):
        to_save = self.state_dict()
        pickle.dump(to_save, open(filename, 'wb'))

    def load(self, filename):
        state_dict = pickle.load(open(filename, 'rb'))
        self.load_state_dict(state_dict)


def train():
    # function called to train the neural network.
    for iteration in range(0, int(max_epoch * data.shape[0] / batch_size)):
        optimizer.zero_grad()

        indices = t.randperm(data.shape[0])[:batch_size]

        train_input = Variable(data[indices])
        train_answers = Variable(targets[indices])

        net_output = model.forward(train_input)

        loss = criterion(net_output, train_answers)

        loss.backward()

        optimizer.step()

        print('epoch = ' + '%.3f' % (iteration * batch_size / data.shape[0]) + ' loss = %.5f' % loss.item())


if __name__ == "__main__":
    os.system('mkdir models')

    # Define the batch size and number of epochs for training.
    batch_size = 15
    max_epoch = 50
    size = 40
    num_data = 1000

    columns = list(np.arange(0, size))

    model = ConvNet(channels1=15, channels2=30, h1=128, h2=64)  # Define the size of the layers, easy to tweak.
    model.training = True
    # print(model.convolution_dimensions(18, 5))

    # file = h5py.File("cooling_history.hdf5", "r")  # Load all the data from the file and organize it into sets.

    dataset = pd.read_csv("/Users/nicholasd./Desktop/git/wip/HRG-Ising/The ANN Test/data/Archive/coolingHistory.csv",
                          usecols=columns, header=None).to_numpy().reshape((num_data, 15, size, size))
    data = t.from_numpy(np.asarray(dataset[()], dtype=np.float32))

    targets = pd.read_csv("/Users/nicholasd./Desktop/git/wip/HRG-Ising/The ANN Test/data/Archive/magField.csv",
                          usecols=columns, header=None).to_numpy().reshape(num_data, size, size)[:,0,0]
    targets = t.from_numpy(np.asarray(targets[()], dtype=np.float32))

    testset = pd.read_csv("/Users/nicholasd./Desktop/git/wip/HRG-Ising/The ANN Test/data/Archive/TestData.csv",
                          usecols=columns, header=None).to_numpy().reshape((int(0.2 * num_data), 15, size, size))
    testset = t.from_numpy(np.asarray(testset[()], dtype=np.float32))

    test_targets = pd.read_csv("/Users/nicholasd./Desktop/git/wip/HRG-Ising/The ANN Test/data/Archive/TestFields.csv",
                               usecols=columns, header=None).to_numpy().reshape((int(0.2 * num_data), size, size))[:,0,0]

    test_targets = t.from_numpy(np.asarray(test_targets[()], dtype=np.float32))

    # Define the loss function to be used and the optimizer, could try SGD.
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.92)

    model.load("models/Lattice_test")  # Load the model. You can train the model a bit, then change the learning rate
    # and load the model to keep training it but with a different learning rate.

    train()  # Train the model.

    # These next lines let you visualize one example from the test set. Change the testindex to get a different sample.
    f, ax = plt.subplots(1, 5)

    testindex = 43
    for i in range(5):
        ax[i].imshow(testset[testindex, i], cmap='winter')

    plt.show()

    f2, ax2 = plt.subplots(1, 2)

    ax2[0].imshow(test_targets[testindex])

    out = model(testset[testindex].unsqueeze(0))
    ax2[1].imshow(out.detach().numpy().squeeze(0))

    plt.show()


# Gives an average loss for the test set to see if overfitting is occuring.
def eval_conv():
    correct = 0
    total = 0

    for image in range(testset.shape[0]):
        label = test_targets[image]

        output = model(testset[image].unsqueeze(0))
        total += criterion(output.squeeze(0), test_targets[image])

    average_loss = total / testset.shape[0]
    print("Average Test Loss: ", average_loss.item())


eval_conv()

model.save("models/Lattice_test")
