import matplotlib
import numpy as np
from Neural_network import NeuralNetwork, NeuralNetwork2

matplotlib.get_backend()
matplotlib.use('QtAgg')


# Define the model parameters
weight = np.array([0.5])
bias = 0.2
learning_rate = 0.01
epochs = 8

# Training data
input_data = np.array([1.0, 2.0, 3.0, 4.0])
desired_output = np.array([num * 3 + 3 for num in input_data])  # 3x + 3

test_input = 5.0
test_prediction = 3 * test_input + 3

# Create and train the neural network
nn = NeuralNetwork2(weight, bias, learning_rate)
nn.train(input_data, desired_output, epochs, test_input, test_prediction)
