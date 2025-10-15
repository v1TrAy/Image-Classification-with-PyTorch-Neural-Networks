# Computational Intelligence - Neural Network Image Classification

This project implements two neural network models using the [translate:torch] library for image classification tasks on a dataset. The goal is to explore different network architectures and hyperparameters to achieve optimal classification accuracy.

## Project Overview

The project focuses on building and training two neural networks:

- The first network has a simple architecture with one hidden layer containing 32 neurons.
- The second network incorporates convolutional layers with pooling to better exploit the spatial structure of image data.

Key aspects of the project include defining network parameters, implementing forward pass, adjusting learning rates, and optimizing using the Adam optimizer with CrossEntropy loss.

## Features

- Support for adjustable batch sizes, learning rates, and epochs.
- A fundamental dense neural network with one hidden layer (32 neurons).
- A convolutional neural network with multiple layers including Conv2D, Pooling, and Flatten layers.
- Use of Adam optimizer for training.
- Evaluation metrics including accuracy.
- Code organization with modular Python scripts for easier maintenance.

## Files in the Repository

- `neuralnet_part1.py`: Basic neural network implementation with one hidden layer.
- `neuralnet_part2.py`: Advanced neural network implementation with convolutional layers.
- `mp5.py`, `reader.py`, `utils.py`: Utility scripts for managing data loading, preprocessing, and training.
- `Report_983212093.pdf`: Project report detailing architecture, hyperparameters, training results, and analysis.

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/computational-intelligence.git
cd computational-intelligence

text

2. Install required packages:

pip install torch torchvision numpy matplotlib

text

## Usage

- Run `neuralnet_part1.py` to train and test the basic neural network.
- Run `neuralnet_part2.py` to train and test the convolutional neural network.
- Use utility scripts for data preparation and evaluation.
- Refer to the notebook or report for detailed instructions and results interpretation.

## Results

- The simple neural network achieved accuracy up to 71% with 128 batch size and learning rate 0.01.
- The convolutional network showed significant improvements by leveraging convolution and pooling layers.
- Optimal configurations involved tuning kernel sizes, learning rates, and batch sizes.
- Training time increased with network complexity, but accuracy and performance improved noticeably.
