import numpy as np
from random import randint
from Neural_Network_n_in_n_out import correct_output, NeuralNetwork_n_n

number_of_neurons = 3
number_of_learn_data = 10
number_of_outputs = 2
weights = np.array([randint(0, 10) / 10 for _ in range(number_of_neurons)])
print(weights)

bias = 0.2
learning_rate = 0.001
epochs = 500

input_data = np.array([[randint(1, 100) / 100 for _ in range(number_of_neurons)] for __ in range(number_of_learn_data)])
print(f"Input data: {input_data}")

multipliers = [randint(1, 10) / 10 for _ in range(number_of_neurons)]
print(f"Multipliers data: {multipliers}")
constant = randint(0, 3)
print(f"constant data: {constant}")

desired_output = correct_output(input_data, multipliers, constant)

test_input = np.array([[randint(1, 100) / 100 for _ in range(number_of_neurons)]])
print(f"test_input data: {test_input}")

test_prediction = correct_output(test_input, multipliers, constant)
print(f"test_prediction data: {test_prediction}")

# Create and train the neural network
nn = NeuralNetwork_n_n(weights, bias, learning_rate, number_of_outputs)  # Pass the number of outputs
nn.train(input_data, desired_output, epochs, test_input, test_prediction, verbose=2, error_percentage=1)
