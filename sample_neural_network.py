import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias2 = np.zeros((1, self.output_size))
        
    def forward(self, X):
        # Forward propagation
        self.hidden_layer = np.dot(X, self.weights1) + self.bias1
        self.hidden_activation = self.sigmoid(self.hidden_layer)
        self.output_layer = np.dot(self.hidden_activation, self.weights2) + self.bias2
        self.output_activation = self.sigmoid(self.output_layer)
        return self.output_activation
        
    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function
        return self.sigmoid(x) * (1 - self.sigmoid(x))



# Specify the input, hidden, and output layer sizes
input_size = 2
hidden_size = 4
output_size = 1

# Create an instance of the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Define your training data (X) and target labels (y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the number of training iterations
num_iterations = 10000

# Define the learning rate
learning_rate = 0.1

# Train the neural network
for i in range(num_iterations):
    # Forward propagation
    output = nn.forward(X)
    
    # Backpropagation
    error = y - output
    d_output = error * nn.sigmoid_derivative(output)
    error_hidden = d_output.dot(nn.weights2.T)
    d_hidden = error_hidden * nn.sigmoid_derivative(nn.hidden_activation)
    
    # Update weights and biases
    nn.weights2 += nn.hidden_activation.T.dot(d_output) * learning_rate
    nn.bias2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    nn.weights1 += X.T.dot(d_hidden) * learning_rate
    nn.bias1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Test the neural network with a new input
new_input = np.array([[0, 1]])
output = nn.forward(new_input)
print(output)

