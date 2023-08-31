import random

import numpy as np


class NeuralNetwork_n_n:
    def __init__(self, weights, biases, learning_rate):
        self.weights = weights
        self.biases = biases
        self.learning_rate = learning_rate
        self.stop = False

    def predict(self, input_data):
        return np.dot(input_data, self.weights) + self.biases

    def stop_test(self, correct_output, prediction, error_percentage):
        errors = np.abs((correct_output - prediction) / correct_output) * 100
        if np.all(errors < error_percentage):
            self.stop = True

    def train(self, input_data, desired_outputs, epochs, test_input, test_prediction, verbose=0, error_percentage=5):
        epoch_list = []
        biases_list = []
        weights_list = []

        for epoch in range(epochs):
            if self.stop:
                break

            epoch_list.append(epoch)

            if verbose == 1:
                print(f"Epoch: {epoch}, prediction: {self.predict(test_input)}")

            if verbose == 2 and (epoch % 10 == 0):
                print(f"Epoch: {epoch}, prediction: {self.predict(test_input)}")

            for i in range(len(input_data)):
                if self.stop:
                    break

                prediction = self.predict(input_data[i])

                error = desired_outputs[i] - prediction

                self.weights += self.learning_rate * np.outer(input_data[i], error)
                self.biases += self.learning_rate * error

            biases_list.append(self.biases)
            weights_list.append(self.weights.copy())  # Store a copy of the weights

            self.stop_test(test_prediction, self.predict(test_input), error_percentage)

        final_prediction = self.predict(test_input)
        print(f"Final Prediction: {final_prediction} vs {test_prediction}")


def correct_output(input_data, multipliers, constant):
    desired_outputs = []
    for i in range(len(input_data)):
        desired_output = np.dot(input_data[i], multipliers) + constant
        desired_outputs.append(desired_output)

    return np.array(desired_outputs)



# TODO  I've undesrtand that I want to make smth unreal
# Example usage
number_of_learn_data = 15
number_of_inputs = 2
number_of_outputs = 1
input_data = np.array([[random.randint(0, 10) / 10 for __ in range(number_of_inputs)] for _ in range(number_of_learn_data)])
print(f"Learn data:{input_data}")

multipliers = [random.randint(1, 10) / 10 for _ in range(number_of_inputs)]
print(f"Multipliers: {multipliers}")
constant = random.randint(0, 3)
print(f"Constant: {constant}")

desired_outputs = correct_output(input_data, multipliers, constant)

print(f"Desired outputs:{desired_outputs}")

test_input = np.array([[random.randint(0, 10) / 10 for _ in range(number_of_inputs)]])
test_prediction = correct_output(test_input, multipliers, constant)

initial_weights = np.random.rand(input_data.shape[1], desired_outputs.shape[1])
initial_biases = np.random.rand(desired_outputs.shape[1])
learning_rate = 0.001
epochs = 500

nn = NeuralNetwork_n_n(initial_weights, initial_biases, learning_rate)
nn.train(input_data, desired_outputs, epochs, test_input, test_prediction, verbose=2)
