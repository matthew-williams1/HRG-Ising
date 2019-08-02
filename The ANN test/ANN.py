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
import h5py


class ConvNet(nn.Module):

    # Image starts at 128x128 for this test.
    def __init__(self, input_channels=5, output_dim=9, channels1=10, channels2=20, h1=200, h2=128, h3=64):
        super(ConvNet, self).__init__()
        self._final = channels2

        self.conv1 = nn.Conv2d(input_channels, channels1, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(channels1, channels2, kernel_size=5, stride=1)
        # self.conv3 = nn.Conv2d(channels2, self._final, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self._final*7*7, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Pooling halves size. Convolution div by 2 and remove 3. After layer: Nx10x18x18.
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # Not sure that pooling is necessarily a good idea after convolutions. Size: Nx20x7x7
        # x = F.relu(self.conv3(x))
        x = x.view(-1, self._final*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 3, 3)
        return x

    def save(self, filename):
        to_save = self.state_dict()
        pickle.dump(to_save, open(filename, 'wb'))

    def load(self, filename):
        state_dict = pickle.load(open(filename, 'rb'))
        self.load_state_dict(state_dict)


def train():
    # function called to train the neural network.
    for iteration in range(0, int(max_epoch*data.shape[0]/batch_size)):

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

    model = ConvNet(channels1=5, channels2=10, h1=128, h2=64)  # Define the size of the layers, easy to tweak.
    model.training = True

    file = h5py.File("cooling_history.hdf5", "r")  # Load all the data from the file and organize it into sets.

    dataset = file["data"]
    data = t.from_numpy(np.asarray(dataset[()], dtype=np.float32))

    targets = file["targets"]
    targets = t.from_numpy(np.asarray(targets[()], dtype=np.float32))

    testset = file["test_data"]
    testset = t.from_numpy(np.asarray(testset[()], dtype=np.float32))

    test_targets = file["test_targets"]
    test_targets = t.from_numpy(np.asarray(test_targets[()], dtype=np.float32))

    # Define the loss function to be used and the optimizer, could try SGD.
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.92)

    model.load("Lattice_test")  # Load the model. You can train the model a bit, then change the learning rate
    # and load the model to keep training it but with a different learning rate.

    # train()  # Train the model.

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

model.save("Lattice_test")
