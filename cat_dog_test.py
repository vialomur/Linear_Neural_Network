import numpy as np
from random import randint
from Neural_network import NeuralNetwork2, correct_output
from data_preprocess import preprocess_image

path = 'data/'

cat_array = preprocess_image(path+"cat.jpg")
dog_array = preprocess_image(path+"dog.jpg")
# Define the model parameters
number_of_neurons = cat_array.shape[0]
weights = np.array([randint(0, 10) / 10 for _ in range(number_of_neurons)])
print(weights)

bias = 0.2
learning_rate = 0.001
epochs = 500

input_data = np.array([cat_array, dog_array])
print(f"Input data: {input_data}")

desired_output = np.array([0, 1])

# Create and train the neural network
nn = NeuralNetwork2(weights, bias, learning_rate)
nn.train(input_data, desired_output, epochs, input_data, desired_output, verbose=2)