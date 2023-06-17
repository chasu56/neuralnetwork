# neuralnetwork

# Neural Network from Scratch


This repository contains a Python implementation of a simple feedforward neural network built from scratch. 
The neural network consists of a single hidden layer and uses the sigmoid activation function.

The sigmoid activation function is a common non-linear activation function used in neural networks. 
It takes an input value and maps it to a value between 0 and 1.


In the code, the nn.forward(new_input) call is using the trained neural network to make a forward pass and generate predictions for the new_input data. 

Here's what happens in the forward method:

The input new_input is multiplied by the first set of weights (self.weights1) and added to the bias term (self.bias1) to compute the hidden layer activations.

The hidden layer activations are then passed through the sigmoid activation function (self.sigmoid) to introduce non-linearity.

The activated hidden layer is multiplied by the second set of weights (self.weights2) and added to the bias term (self.bias2) to compute the output layer activations.

The output layer activations are passed through the sigmoid activation function again to obtain the final output of the neural network.

The returned value from nn.forward(new_input) is the output of the neural network for the given input new_input. It represents the model's prediction based on the trained weights and biases.

## Requirements

- Python 3.10
- NumPy

## Usage

1. Clone the repository:

```bash
git clone https://github.com/chasu56/neuralnetwork.git
