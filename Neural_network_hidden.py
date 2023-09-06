from random import randint
import numpy as np


class InputLayer:
    def __init__(self, input_data, learning_rate):
        self.weights = np.array([randint(0, 100) / 100 for _ in range(len(input_data))])
        self.bias = randint(0, 100) / 100
        self.values = input_data * self.weights
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


class HiddenLayer(InputLayer):
    def __init__(self, prev_layer_values, size, learning_rate):
        self.weights = np.array([randint(0, 100) / 100 for _ in range(size)])
        self.values = prev_layer_values * self.weights
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


class NeuralNetwork3:
    def __init__(self, input_data: np.array, learning_rate: float, output_size: int):
        self.input_layer = InputLayer(input_data, learning_rate)
        self.hidden_layer = HiddenLayer(self.input_layer.values, len(self.input_layer.weights), learning_rate)
        self.learning_rate = learning_rate
        self.output_weights = np.array([randint(0, 100) for _ in range(output_size)])
        self.stop = False

    def stop_test(self, correct_output, prediction, error_percentage):
        epsilon = 1e-9
        if np.all((((abs(correct_output - prediction)) / (correct_output + epsilon)) * 100) < error_percentage):
            self.stop = True

    def predict(self, input_data):
        pass

    def train(self, input_data, desired_output, epochs, test_input, test_prediction, verbose=0, error_percentage=5):
        for epoch in range(epochs):
            if self.stop:
                break

            if verbose == 1:
                print(f"Epoch: {epoch}, prediction: {self.hidden_layer.predict()}")

            if verbose == 2 and (epoch % 10 == 0):
                print(f"Epoch: {epoch}, prediction: {self.hidden_layer.predict()}")

            for i in range(len(input_data)):
                if self.stop:
                    break

                self.input_layer.train_one_input_data(desired_output[i])

                self.hidden_layer.train_one_input_data(desired_output[i])

            self.stop_test(test_prediction, self.hidden_layer.predict(), error_percentage)

        final_prediction = self.hidden_layer.predict()
        print(f"Final Prediction: {final_prediction} vs {test_prediction[0]}")


def correct_output(input_data, multipliers, constant):
    desired_outputs = []
    for i in range(len(input_data)):
        desired_output = np.dot(input_data[i], multipliers) + constant
        desired_outputs.append(desired_output)

    return np.array(desired_outputs)
