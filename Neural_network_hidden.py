from random import randint
import numpy as np


class InputLayer:
    def __init__(self, input_data, learning_rate):
        self.values = input_data
        self.weights = np.array([randint(0, 100) / 100 for _ in range(len(input_data))])
        self.bias = randint(0, 100) / 100
        self.learning_rate = learning_rate
        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = 0
        self.v_bias = 0
        self.t = 0  # Time step

    def predict(self):
        return np.dot(self.values, self.weights) + self.bias

    def update_one_weight(self, input_data, error, j):
        # Compute gradients
        gradient = error * input_data[j]

        # Update first and second moments
        self.m_weights[j] = self.beta1 * self.m_weights[j] + (1 - self.beta1) * gradient
        self.v_weights[j] = self.beta2 * self.v_weights[j] + (1 - self.beta2) * (gradient ** 2)

        # Bias correction
        m_hat = self.m_weights[j] / (1 - self.beta1 ** self.t)
        v_hat = self.v_weights[j] / (1 - self.beta2 ** self.t)

        # Update weights
        self.weights[j] += self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def train_one_input_data(self, desired_output):
        prediction = self.predict()
        error = desired_output - prediction

        self.t += 1  # Increment the time step

        j = 0
        while j < len(self.weights):
            self.update_one_weight(self.values, error, j)
            j += 1


class HiddenLayer:
    def __init__(self, prediction_from_prev_layer, size):
        self.values = np.array([prediction_from_prev_layer for _ in range(size)])
        self.weights = np.array([randint(0, 100) / 100 for _ in range(size)])
        self.bias = randint(0, 100) / 100

    def predict(self):
        return np.dot(self.values, self.weights) + self.bias


class NeuralNetwork3:
    def __init__(self, input_data: np.array, learning_rate: float):
        self.input_layer = InputLayer(input_data, learning_rate)
        self.hidden_layer = HiddenLayer(self.input_layer.predict(),  len(self.input_layer.weights))
        self.learning_rate = learning_rate
        self.stop = False

    def stop_test(self, correct_output, prediction, error_percentage):
        epsilon = 1e-9
        if np.all((((abs(correct_output - prediction)) / (correct_output + epsilon)) * 100) < error_percentage):
            self.stop = True


    def train(self, input_data, desired_output, epochs, test_input, test_prediction, verbose=0, error_percentage=5):
        for epoch in range(epochs):
            if self.stop:
                break

            if verbose == 1:
                print(f"Epoch: {epoch}, prediction: {self.predict(test_input, self.hidden_weights, self.bias_hidden)}")

            if verbose == 2 and (epoch % 10 == 0):
                print(f"Epoch: {epoch}, prediction: {self.predict(test_input, self.hidden_weights, self.bias_hidden)}")

            for i in range(len(input_data)):
                if self.stop:
                    break

                self.train_one_input_data(input_data[i], desired_output[i])

            self.stop_test(test_prediction, self.predict(test_input, self.hidden_weights, self.bias_hidden), error_percentage)

        final_prediction = self.predict(test_input, self.hidden_weights, self.bias_hidden)
        print(f"Final Prediction: {final_prediction} vs {test_prediction[0]}")


def correct_output(input_data, multipliers, constant):
    desired_outputs = []
    for i in range(len(input_data)):
        desired_output = np.dot(input_data[i], multipliers) + constant
        desired_outputs.append(desired_output)

    return np.array(desired_outputs)
