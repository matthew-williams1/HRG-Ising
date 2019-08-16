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

targets = pd.read_csv(
    "/Users/alexbeaudin/Documents/C++/Ising/train/labels.csv", usecols=columns, header=None, dtype=np.float32)\
                       .to_numpy().reshape(num_data, 1, size, size)

val_data = t.from_numpy(pd.read_csv(
    "/Users/alexbeaudin/Documents/C++/Ising/validate/data.csv", usecols=columns, header=None, dtype=np.float32)\
                       .to_numpy().reshape((200, 15, size, size)))

val_targets = pd.read_csv(
    "/Users/alexbeaudin/Documents/C++/Ising/validate/labels.csv", usecols=columns, header=None, dtype=np.float32)\
                            .to_numpy().reshape(200, 1, size, size)

# Modify the targets so that they are comprised of a map which is 1, where there is a field and zeros elsewhere.
# Second part of tuple is the magnitude of the field.
# TODO: Modify this in the data making code if it is successful. Also, experiment with running averages so there
#  might be more info in the input data.
strengths = np.amax(targets, axis=(1, 2, 3)) .reshape(num_data, 1) # The strength of each field is just the max value of the map.
targets = t.from_numpy(targets / strengths.reshape(num_data, 1, 1, 1))  # Map divided by strength is the same shape but 1s where the field is.
strengths = t.from_numpy(strengths)
val_strengths = np.amax(val_targets, axis=(1, 2, 3)).reshape(200, 1)  # Same for the validation set.
val_targets = t.from_numpy(val_targets / val_strengths.reshape(200, 1, 1, 1))
val_strengths = t.from_numpy(val_strengths)


# Start by defining the network.
class ConvNet(nn.Module):

    def __init__(self, in_features=15, h1=300, h2=150):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_features, 15, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(15, 1, kernel_size=3, padding=1, bias=True)

        self.activation = nn.Tanh()
        self.activation2 = nn.Tanhshrink()

        self.fc1 = nn.Linear(1, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

        self.pool = nn.AdaptiveAvgPool3d((1, 40, 40))

        self.normalize = nn.BatchNorm2d(15)

    def forward(self, input):
        """
        Forward method for the neural network.
        :param input: Input is a Nx15x40x40 array which corresponds to the cooling history of N lattices where N is
        the batch size.
        :return: A tuple which contains a probability field for the location of the magnetic field in the first
        position, and a real number which corresponds to the magnitude as the second number.
        """
        x = (t.mul(self.conv1(input), input))
        x = self.activation(self.conv2(x.unsqueeze(1)))
        x = F.hardtanh(self.conv3(x.squeeze(1)), min_val=0, max_val=1)
        x = x.view(-1, 1, 40, 40)

        processed = input * x[:, :, :, :]
        strength = t.sum(processed, dim=(1, 2, 3)) / t.sum(x, dim=(1, 2, 3))
        strength = strength.view(-1, 1)
        strength = self.activation(self.fc1(strength))
        strength = self.activation(self.fc2(strength))
        strength = self.fc3(strength)

        return x, strength

    def save(self, filename):
        to_save = self.state_dict()
        pickle.dump(to_save, open(filename, 'wb'))

    def load(self, filename):
        state_dict = pickle.load(open(filename, 'rb'))
        self.load_state_dict(state_dict)


if __name__ == '__main__':

    # Set the parameters for training.
    max_epochs = 3

    model = ConvNet(in_features=15, h1=500, h2=250)

    criterion1 = nn.MSELoss(reduction='mean')
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    model.load("Recognition")  # Load the model. You can train the model a bit, then change the learning rate
    # and load the model to keep training it but with a different learning rate.

    epochs = []
    training_loss = []
    validate_loss = []

    if True:
        record = np.inf
        increase_count = 0
        increase_threshold = 6
        
        for iteration in range(int(max_epochs * 1000 / BATCH_SIZE)):

            if iteration - int(iteration) < 0.001:
                shuffled = t.randperm(num_data)

            model = model.train()

            start_index = (iteration % int(1000 / BATCH_SIZE)) * BATCH_SIZE
            indices = shuffled[start_index:start_index+BATCH_SIZE]

            input = Variable(data[indices])
            label = (targets[indices], strengths[indices])

            prediction = model(input)
            train_loss_shape = criterion1(prediction[0], label[0])
            train_loss_strength = criterion2(prediction[1], label[1])
            train_loss = train_loss_shape + train_loss_strength
            training_loss.append(train_loss)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            model = model.eval()
            prediction = model(val_data)
            val_loss_shape = criterion1(prediction[0], val_targets)
            val_loss_strength = criterion2(prediction[1], val_strengths)
            val_loss = val_loss_shape + val_loss_strength
            validate_loss.append(val_loss)
            
            if record > val_loss:
                record = val_loss
                model.save("models/Recognition")

            if len(validate_loss) > 1 and val_loss < validate_loss[-2]:
                increase_count = 0
            else:
                increase_count += 1

            epochs.append(iteration * BATCH_SIZE / 1000)
            print("Epoch %.3f/%i; Train loss: %.3f; Validation loss: %.3f."
              % (iteration * BATCH_SIZE / 1000, max_epochs, train_loss.item(), val_loss.item()))

            if increase_count >= increase_threshold:
                break

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

    out = model(input.unsqueeze(0))

    plt.subplot(211)
    plt.imshow(label.detach().numpy()[0])
    plt.subplot(212)
    plt.imshow(out[0].detach().numpy()[0, 0])

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()

    print("Actual Magnitude: %.3f, Predicted Magnitude: %.3f."% (val_strengths[index].item(), out[1].item()))

    plt.show()
