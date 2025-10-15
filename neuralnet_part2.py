# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester

"""
This is the main entry point for MP5. You should only modify code
within this file, neuralnet_learderboard and neuralnet_part1.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

       

        """
         # Number of hidden layer units: 100 as per architecture specification
        h = 100
        
        # Convolution layer parameters
        conv_out = 32  # Number of output channels after the convolution layer
        conv_kernel = 5  # Size of the kernel for the convolution layer
        pool_kernel = 5  # Size of the kernel for the pooling layer

        # Dimensions of the input images (assumed square images)
        img_h = 31
        img_w = 31
        img_c = 3  # Number of image channels (e.g., 3 for RGB)

        # Call the parent class's constructor
        super(NeuralNet, self).__init__()
        
        # Store the specified loss function
        self.loss_fn = loss_fn

        # Image dimension attributes
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c

        # Neural network architecture using Sequential container
        self.network = nn.Sequential(
            nn.Conv2d(img_c, conv_out, conv_kernel),  # Convolution layer
            nn.ReLU(),  # Activation layer
            nn.MaxPool2d(pool_kernel),  # Pooling layer
            nn.Flatten(),  # Flattening layer to convert 2D features to 1D for fully connected layers
            nn.Linear(
                # Calculate the flattened size after conv and pooling layers
                conv_out * ((img_h - conv_kernel + 1) // pool_kernel) *
                ((img_w - conv_kernel + 1) // pool_kernel),
                h  # Number of hidden units
            ),
            nn.ReLU(),  # Activation layer
            nn.Linear(h, out_size)  # Output linear layer with out_size units
        )

        # Initialize the optimizer for training (use Stochastic Gradient Descent)
        self.optimizer = optim.SGD(self.parameters(), lrate)

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        return self.network(
            x.reshape(
                (-1, self.img_c, self.img_h, self.img_w)
            )
        )

    def step(self, x: torch.Tensor, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
# Reset gradients in the optimizer
        self.optimizer.zero_grad()

        # Forward pass (compute predictions)
        yhat = self.forward(x)

        # Compute loss by comparing predictions with true labels
        loss_value = self.loss_fn(yhat, y)

        # Backward pass (compute gradients)
        loss_value.backward()

        # Update weights based on computed gradients
        self.optimizer.step()

        # Return the loss value to track progress
        return loss_value.item()


def fit(train_set, train_labels, dev_set, epochs, batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
# Initialize the learning rate
learning_parameter = 0.1

# Normalize function to apply on data sets
def normalize_data(data_set):
    data_deviation, data_ave = torch.std_mean(data_set)
    return (data_set - data_ave) / data_deviation

# Normalize training and validation sets
normalized_train = normalize_data(train_set)
normalized_dev = normalize_data(dev_set)

# Setup for iterating over batches
prepared_data = get_dataset_from_arrays(normalized_train, train_labels)
batch_processor = DataLoader(prepared_data, batch_size)

# Configure the neural network and keep track of loss records
net = NeuralNet(learning_parameter, nn.CrossEntropyLoss(), len(normalized_train[0]), 4)
losses = []

# Begin the training process
print('Initiating training phase...')
for current_epoch in range(epochs):
    cumulative_loss = 0.0
    for batch_data in batch_processor:
        training_loss = net.step(batch_data['features'], batch_data['labels'])
        cumulative_loss += training_loss
    losses.append(cumulative_loss)
    print(f'\rCurrent Epoch: {current_epoch + 1}/{epochs}', end='')
print('\nTraining phase complete')

# Assessment using the dev set
print('Conducting evaluation on dev set...')
yhats = np.argmax(net.forward(normalized_dev).cpu().numpy(), axis=1)
print('Evaluation phase complete')

# Outcome of the function
return losses, yhats, net