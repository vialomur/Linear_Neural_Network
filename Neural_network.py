import random

import numpy as np


class Node:
    def __init__(self, value):
        self.value = value
        self.weight = random.randint(1, 10) / 10


class InputLayer:
    def __init__(self, values):
        self.nodes = []
        for value in values:
            self.nodes.append(Node(value))

        self.nodes = np.array(self.nodes)


class NeuralNetwork2:
    def __init__(self, weights, bias, learning_rate):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.stop = False

    def predict(self, input_data):
        return sum(input_data[0] * self.weights) + self.bias

    def stop_test(self, correct_output, prediction, error_percentage):
        if (((abs(correct_output - prediction)) / correct_output) * 100) < error_percentage:
            self.stop = True

    def train(self, input_data, desired_output, epochs, test_input, test_prediction, verbose=0, error_percentage=5):
        epoch_list = []
        error_list = []
        bias_list = []
        weight_list = []

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

                error = desired_output[i] - prediction

                for j in range(len(self.weights)):
                    self.weights[j] += self.learning_rate * error * input_data[i][j]

                self.bias += self.learning_rate * error

            # total_error = abs(test_prediction - prediction) TODO fix error appending
            # error_list.append(total_error)
            bias_list.append(self.bias)
            weight_list.append(self.weights[:])  # Store a copy of the weights
            self.stop_test(test_prediction, self.predict(test_input), error_percentage)

        final_prediction = self.predict(test_input)
        print(f"Final Prediction: {final_prediction} vs {test_prediction[0]}")


def correct_output(input_data, multipliers, constant):
    desired_outputs = []
    for i in range(len(input_data)):
        desired_output = np.dot(input_data[i], multipliers) + constant
        desired_outputs.append(desired_output)

    return np.array(desired_outputs)
