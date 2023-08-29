import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork:
    def __init__(self, weight, bias, learning_rate):
        self.weight = weight
        self.bias = bias
        self.learning_rate = learning_rate

    def predict(self, input_data):
        return input_data * self.weight + self.bias

    def train(self, input_data, desired_output, epochs, test_input, test_prediction, verbose=0):
        epoch_list = []
        error_list = []
        bias_list = []
        weight_list = []

        for epoch in range(epochs):
            epoch_list.append(epoch)

            if verbose:
                print(f"Epoch: {epoch}, prediction: {self.predict(test_input)}")

            prediction = 0
            for i in range(len(input_data)):
                prediction = self.predict(input_data[i])

                error = desired_output[i] - prediction

                self.weight += self.learning_rate * error * input_data[i]
                self.bias += self.learning_rate * error

            total_error = abs(test_prediction - prediction)
            error_list.append(total_error)
            bias_list.append(self.bias)
            weight_list.append(self.weight)

        final_prediction = self.predict(test_input)
        print("Final Prediction:", final_prediction)

    def plot_learning_process(self, epochs, error_list, bias_list, weight_list):
        plt.figure(figsize=(10, 5))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, error_list, marker='o')
        plt.title('Error vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Error')

        plt.subplot(2, 1, 2)
        plt.plot(epochs, bias_list, label='Bias', marker='o')
        plt.plot(epochs, weight_list, label='Weight', marker='o')
        plt.title('Bias and Weight vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()

        plt.tight_layout()
        plt.show()


class NeuralNetwork2:
    def __init__(self, weights, bias, learning_rate):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

    def predict(self, input_data):
        return sum(input_data[0] * self.weights) + self.bias

    def train(self, input_data, desired_output, epochs, test_input, test_prediction, verbose=0):
        epoch_list = []
        error_list = []
        bias_list = []
        weight_list = []

        for epoch in range(epochs):
            epoch_list.append(epoch)

            if verbose:
                print(f"Epoch: {epoch}, prediction: {self.predict(test_input)}")

            prediction = 0
            for i in range(len(input_data)):
                prediction = self.predict(input_data[i])

                error = desired_output[i] - prediction

                for j in range(len(self.weights)):
                    self.weights[j] += self.learning_rate * error * input_data[i][j]

                self.bias += self.learning_rate * error

            total_error = abs(test_prediction - prediction)
            error_list.append(total_error)
            bias_list.append(self.bias)
            weight_list.append(self.weights[:])  # Store a copy of the weights

        final_prediction = self.predict(test_input)
        print("Final Prediction:", final_prediction)


def correct_output(input_data, multipliers, constant):
    desired_outputs = []
    for i in range(len(input_data)):
        desired_output = 0
        for j in range(len(input_data[0])):
            desired_output += input_data[i][j] * multipliers[j]
        desired_output += constant
        desired_outputs.append(desired_output)

    return np.array(desired_outputs)
