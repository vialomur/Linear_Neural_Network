import numpy as np
from random import randint
from Neural_network import NeuralNetwork2, correct_output

# Define the model parameters
number_of_neurons = 3
number_of_input_data = 4
weights = np.array([randint(0, 10) / 10 for _ in range(number_of_neurons)])
print(weights)

bias = 0.2
learning_rate = 0.01
epochs = 25

input_data = np.array([[randint(1, 100) / 100 for _ in range(number_of_neurons)] for __ in range(number_of_input_data)])
print(f"Input data: {input_data}")

multipliers = [randint(1, 10) / 10 for _ in range(number_of_neurons)]
print(f"Multipliers data: {multipliers}")
constant = randint(1, 5)
print(f"constant data: {constant}")

desired_output = correct_output(input_data, multipliers, constant)

test_input = np.array([[randint(1, 100) / 100 for _ in range(number_of_neurons)]])
print(f"test_input data: {test_input}")

test_prediction = correct_output(test_input, multipliers, constant)
print(f"test_prediction data: {test_prediction}")

# Create and train the neural network
nn = NeuralNetwork2(weights, bias, learning_rate)
nn.train(input_data, desired_output, epochs, test_input, test_prediction)
