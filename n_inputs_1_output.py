import numpy as np

# Define the model parameters
from Neural_network import NeuralNetwork2

weights = [0.5, 0.3]  # Two weights for two inputs
bias = 0.2
learning_rate = 0.01
epochs = 8

# Training data (two inputs for each data point)
input_data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
desired_output = np.array([num[0] * 3 + num[1] * 2 + 3 for num in input_data])  # 3x1 + 2x2 + 3

test_input = np.array([5.0, 6.0])  # Two inputs for testing
test_prediction = np.array(test_input[0] * 3 + test_input[1] * 2 + 3)

# Create and train the neural network
nn = NeuralNetwork2(weights, bias, learning_rate)
nn.train(input_data, desired_output, epochs, test_input, test_prediction)
