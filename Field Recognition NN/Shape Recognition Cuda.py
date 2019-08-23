import numpy as np
import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from matplotlib.ticker import PercentFormatter

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
    "train/data.csv", usecols=columns, header=None, dtype=np.float32)\
                    .to_numpy().reshape((num_data, 15, size, size))).cuda()

targets = pd.read_csv(
    "train/labels.csv", usecols=columns, header=None, dtype=np.float32)\
                       .to_numpy().reshape(num_data, 1, size, size)

val_data = t.from_numpy(pd.read_csv(
    "validate/data.csv", usecols=columns, header=None, dtype=np.float32)\
                       .to_numpy().reshape((200, 15, size, size))).cuda()

val_targets = pd.read_csv(
    "validate/labels.csv", usecols=columns, header=None, dtype=np.float32)\
                            .to_numpy().reshape(200, 1, size, size)

# Modify the targets so that they are comprised of a map which is 1, where there is a field and zeros elsewhere.
# Second part of tuple is the magnitude of the field.
# TODO: Modify this in the data making code if it is successful. Also, experiment with running averages so there
#  might be more info in the input data.
strengths = np.amax(targets, axis=(1, 2, 3)) .reshape(num_data, 1) # The strength of each field is just the max value of the map.
targets = t.from_numpy(targets / strengths.reshape(num_data, 1, 1, 1)).cuda()  # Map divided by strength is the same shape but 1s where the field is.
strengths = t.from_numpy(strengths).cuda()
val_strengths = np.amax(val_targets, axis=(1, 2, 3)).reshape(200, 1)  # Same for the validation set.
val_targets = t.from_numpy(val_targets / val_strengths.reshape(200, 1, 1, 1)).cuda()
val_strengths = t.from_numpy(val_strengths).cuda()


# Start by defining the network.
class ConvNet(nn.Module):

    def __init__(self, in_features=15, h1=300, h2=150):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(15*8, 8, kernel_size=3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(8, 1, kernel_size=3, padding=1, padding_mode='wrap')

        self.activation = nn.Tanh()
        self.activation2 = nn.Tanhshrink()

        self.fc1 = nn.Linear(1, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

        self.fc4 = nn.Linear(64, 1)

        self.pool = nn.AdaptiveAvgPool3d((3, 10, 10))

        self.normalize = nn.BatchNorm2d(15)

    def forward(self, input):
        """
        Forward method for the neural network.
        :param input: Input is a Nx15x40x40 array which corresponds to the cooling history of N lattices where N is
        the batch size.
        :return: A tuple which contains a probability field for the location of the magnetic field in the first
        position, and a real number which corresponds to the magnitude as the second number.
        """
        x = self.activation(self.conv1(input.unsqueeze(1)))
        x = x.view(-1, 120, 40, 40)
        x = self.activation(self.conv3(x))
        #strength = x.view(-1, 8, 40, 40)
        x = self.conv5(x)
        x = t.div(x, t.max(x.max(dim=2, keepdim=True)[0], dim=3, keepdim=True)[0])
        x = F.relu(x)
        x = x.view(-1, 1, 40, 40)

        with t.no_grad():
            map = x * input
            sum = t.sum(map, dim=(1, 2, 3)) / t.sum(x, dim=(1, 2, 3))
            sum = sum.view(-1, 1)

        strength = self.activation(self.fc1(sum))
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
    max_epochs = 20

    model = ConvNet(in_features=15, h1=512, h2=256)
    model = model.cuda()

    criterion1 = nn.MSELoss(reduction='mean')
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model.load("Recognition3d")  # Load the model. You can train the model a bit, then change the learning rate
    # and load the model to keep training it but with a different learning rate.

    epochs = []
    training_loss = []
    validate_loss = []

    if False:
        record = np.inf
        increase_count = 0
        increase_threshold = 6

        for iteration in range(int(max_epochs * 1000 / BATCH_SIZE)):

            if iteration - int(iteration) < 0.001:
                shuffled = t.randperm(num_data)

            model = model.train()

            start_index = (iteration % int(1000 / BATCH_SIZE)) * BATCH_SIZE
            indices = shuffled[start_index:start_index+BATCH_SIZE]

            input = data[indices]
            label = (targets[indices], strengths[indices])

            prediction = model(input)
            train_loss_shape = criterion1(prediction[0], label[0])
            train_loss_strength = criterion2(prediction[1], label[1])
            train_loss = train_loss_strength
            training_loss.append(train_loss.item())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            model = model.eval()
            prediction = model(val_data)
            val_loss_shape = criterion1(prediction[0], val_targets)
            val_loss_strength = criterion2(prediction[1], val_strengths)
            val_loss = val_loss_strength
            validate_loss.append(val_loss.item())

            if record > val_loss.item():
                record = val_loss.item()
                model.save("Recognition3d")

            if len(validate_loss) > 1 and val_loss < validate_loss[-2]:
                increase_count = 0
            else:
                increase_count += 1

            epochs.append(iteration * BATCH_SIZE / 1000)
            print("Epoch %.3f/%i; Train loss: %.3f; Validation loss: %.3f."
              % (iteration * BATCH_SIZE / 1000, max_epochs, train_loss.item(), val_loss.item()))

            if increase_count >= increase_threshold:
                break

            t.cuda.empty_cache()

    plt.plot(epochs, training_loss)
    plt.plot(epochs, validate_loss)
    plt.show()

    model = model.eval()
    model = model.cpu()
    index = np.random.randint(0, 200)
    if np.abs(val_strengths[index].cpu() - 5.166) > 0.001:
        index = np.random.randint(0, 200)

    input = val_data[index].cpu()
    label = val_targets[index].cpu()

    f, ax = plt.subplots(3, 5)
    frame = 0
    for k in range(3):
        for i in range(5):
            ax[k, i].imshow(input[frame].detach().numpy(), cmap='winter')
            frame += 1


    plt.show()

    out = model(input.unsqueeze(0))

    plt.subplot(211)
    plt.imshow(label.detach().numpy()[0], cmap='plasma')
    plt.subplot(212)
    plt.imshow(out[0].detach().numpy()[0, 0], cmap='plasma')

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)

    print("Actual Magnitude: %.3f, Predicted Magnitude: %.3f." % (val_strengths[index].item(), out[1].item()))

    plt.show()
    out = model(val_data.cpu())[0]
    xdata = (((t.sum(out * val_data.cpu(), dim=(1, 2, 3)) / t.sum(out, dim=(1, 2, 3))).cpu() / 15) + 1) / 2

    true_data = (((t.sum(val_data * val_targets, dim=(1, 2, 3)) / t.sum(val_targets, dim=(1, 2, 3))).cpu() / 15) + 1)/2
    plt.plot(xdata.detach().numpy(), val_strengths.cpu().detach().numpy(), 'b.', label='Sum using maps made by NN')
    plt.plot(true_data.cpu().detach().numpy(), val_strengths.cpu().detach().numpy(), 'g.', label='Sum using answer maps')
    plt.plot(xdata.detach().numpy(), model(val_data.cpu())[1].detach().numpy(), 'r.', label='Curve made by NN using NN maps')
    plt.xlabel("Upspin Proportion")
    plt.ylabel("Field Strength")
    plt.title("Field strength as a function of up-spin proportion")
    plt.legend()
    plt.show()

    out = model(val_data.cpu())[1].detach().numpy()
    val_strengths = val_strengths.cpu().detach().numpy()

    error = np.abs(val_strengths - out) / val_strengths * 100
    plt.plot(out, error, 'b.')
    plt.xlabel("Predicted Magnitude")
    plt.ylabel("Percent Error (%)")
    plt.title("Percent Error as a Function of Predicted Output")
    plt.show()

    fig, axs = plt.subplots(1, 1)
    axs.hist(error, bins=10, range=(0, 50))
    axs.yaxis.set_major_formatter(PercentFormatter(xmax=error.shape[0]))
    plt.show()

    np.save("Error.npy", error)
    np.save("Predictions.npy", out)
