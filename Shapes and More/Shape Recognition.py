import numpy as np
import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import pandas as pd
import sys

'''
The purpose of this file is only to see if a network can learn
to recognize the shape of a magnetic field. Ideally,
the neural network will be as simple as possible. Additionally, 
the data was curated such that the strength of the 
magnetic field is always 5; however, the shape changes.
'''

BATCH_SIZE = 32
size = 40
num_data = 1000
columns = list(np.arange(0, size))

data = t.from_numpy(pd.read_csv(
    "/Users/alexbeaudin/Documents/C++/Ising/train/data.csv", usecols=columns, header=None, dtype=np.float32)\
                    .to_numpy().reshape((num_data, 15, size, size)))

targets = t.from_numpy(pd.read_csv(
    "/Users/alexbeaudin/Documents/C++/Ising/train/labels.csv", usecols=columns, header=None, dtype=np.float32)\
                       .to_numpy().reshape(num_data, 1, size, size))

val_data = t.from_numpy(pd.read_csv(
    "/Users/alexbeaudin/Documents/C++/Ising/validate/data.csv", usecols=columns, header=None, dtype=np.float32)\
                       .to_numpy().reshape((200, 15, size, size)))

val_targets = t.from_numpy(pd.read_csv(
    "/Users/alexbeaudin/Documents/C++/Ising/validate/labels.csv", usecols=columns, header=None, dtype=np.float32)\
                            .to_numpy().reshape(200, 1, size, size))


# Start by defining the network.
class ConvNet(nn.Module):

    def __init__(self, in_features=15, h1=300, h2=150):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_features, 15, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(15, 1, kernel_size=3, padding=1, bias=True)

        self.activation = nn.Tanh()
        self.activation2 = nn.Tanhshrink()

        self.fc1 = nn.Linear(2*40*40, h1)
        self.fc2 = nn.Linear(h1, 40*40)

        self.pool = nn.AdaptiveAvgPool3d((1, 40, 40))

        self.normalize = nn.BatchNorm2d(15)

    def forward(self, x):
        #TODO: Possibly add a layer which finds the strength, and multiply a convolution by that strength

        x = (t.mul(self.conv1(x), x))
        #x = self.normalize(x)
        x = self.activation(self.conv2(x.unsqueeze(1)))
        x = self.activation(self.conv3(x.squeeze(1)))
        x = x.view(-1, 1, 40, 40)

        return x

    def save(self, filename):
        to_save = self.state_dict()
        pickle.dump(to_save, open(filename, 'wb'))

    def load(self, filename):
        state_dict = pickle.load(open(filename, 'rb'))
        self.load_state_dict(state_dict)


if __name__ == '__main__':

    # Set the parameters for training.
    max_epochs = 5

    model = ConvNet(in_features=15, h1=4*40*40, h2=2*40*40)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    #model.load("Recognition")  # Load the model. You can train the model a bit, then change the learning rate
    # and load the model to keep training it but with a different learning rate.

    epochs = []
    training_loss = []
    validate_loss = []

    if True:
        for iteration in range(int(max_epochs * 1000 / BATCH_SIZE)):

            if iteration - int(iteration) < 0.001:
                shuffled = t.randperm(num_data)

            model = model.train()

            start_index = (iteration % int(1000 / BATCH_SIZE)) * BATCH_SIZE
            indices = shuffled[start_index:start_index+BATCH_SIZE]

            input = Variable(data[indices])
            label = Variable(targets[indices])

            train_loss = criterion(model(input), label)
            training_loss.append(train_loss)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            model = model.eval()

            val_loss = criterion(model(val_data), val_targets)
            validate_loss.append(val_loss)

            epochs.append(iteration * BATCH_SIZE / 1000)
            print("Epoch %.3f/%i; Train loss: %.3f; Validation loss: %.3f."
              % (iteration * BATCH_SIZE / 1000, max_epochs, train_loss.item(), val_loss.item()))

    model.save("Recognition")

    plt.plot(epochs, training_loss)
    plt.plot(epochs, validate_loss)
    plt.show()

    model = model.eval()
    index = np.random.randint(0, 200)
    input = val_data[index]
    label = val_targets[index]

    f, ax = plt.subplots(3, 5)
    frame = 0
    for k in range(3):
        for i in range(5):
            ax[k, i].imshow(input[frame].detach().numpy(), cmap='winter')
            frame += 1

    plt.show()

    f2, ax2 = plt.subplots(1, 2)
    ax2[0].imshow(label.detach().numpy()[0])

    out = model(input.unsqueeze(0))
    out = out.detach().numpy()

    ax2[1].imshow(out[0, 0])

    plt.show()
