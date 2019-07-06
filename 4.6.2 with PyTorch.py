import torch
from torch.autograd import Variable
from torch.optim import SGD
import torch.nn as NN
import torch.nn.functional as F
import numpy as np
import random


# Determines the dimensionality of inputs and the number of data samples.
dimensions = 10
num_samples = 100

true_coeff = np.random.normal(0, 2, dimensions)  # Make coefficients for f(x).
true_coeff = torch.tensor(true_coeff, dtype=torch.float)
# Coefficients pulled from a normal dist mean 0 s.d. 2
intercept = random.uniform(-5, 5)  # Create random value from -5,5 for intercept.


def lin_function(x):
    # Actual function which determines outputs from inputs, used to create training data.
    return torch.sum(true_coeff * x) + intercept


def make_test_data(f, m, samples):
    # Create and organize data used as samples.
    data_x = torch.zeros([samples, m])  # Using np tensors to manage data.
    data_y = torch.zeros([samples])

    for i in range(samples):
        for j in range(m):  # Go through all slots in arrays to generate random numbers in x
            data_x[i][j] = random.randint(0, 100)

        # Compute exact outputs of test samples given the function.
        # Add a bit of noise to outputs to not get exact values of coefficients.
        data_y[i] = f(data_x[i]) + np.random.normal(0, 5)

    # There is chance of inconsistency. With m = 5 and 100 samples, the prob is ~1.3e-6.
    return data_x, data_y


# Get samples and organize them.
train_samples, train_answers = make_test_data(lin_function, dimensions, num_samples)

# Turn training samples into torch Variable to keep track of gradient.
train_samples = Variable(train_samples, requires_grad=True)

f1 = NN.Linear(dimensions, 1)  # Torch linear layer which takes an input of size dimensions, and returns out of size 1.

model = NN.Sequential(  # Make a sequential model which has only a fully connected linear layer.
        f1
    )

optim = SGD(model.parameters(), lr=5e-6, momentum=0.9)  # Set up optimizer for the model. (Stochastic Gradient Descent)


'''Now time to train the model, could have been done in a separate function.'''
for epoch in range(1, 101):

    predictions = model(train_samples)  # Get the output from the linear layer

    loss = F.l1_loss(torch.squeeze(predictions), train_answers, reduction='sum')  # Calculate loss

    optim.zero_grad()  # Clear the gradients of the optimizer.
    # (Reset to zero, since optimizers in PyTorch accumulate gradients).

    loss.backward()  # Compute the gradients at every operation from the loss function.

    optim.step()  # Take a small step.

    if epoch % 20 == 0:
        print("Epoch: %d; Loss: %f" % (epoch, loss.item()))


print("True coefficients: ", true_coeff)
print("Intercept: ", intercept)

print("Solved coefficients: ", f1._parameters)

