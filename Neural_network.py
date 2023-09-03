import multiprocessing

import numpy as np


class NeuralNetwork2:
    def __init__(self, weights, bias, learning_rate):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.stop = False
        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_weights = np.zeros_like(weights)
        self.v_weights = np.zeros_like(weights)
        self.m_bias = 0
        self.v_bias = 0
        self.t = 0  # Time step

    def predict(self, input_data):
        return np.dot(input_data, self.weights) + self.bias

    def stop_test(self, correct_output, prediction, error_percentage):
        epsilon = 1e-9
        if np.all((((abs(correct_output - prediction)) / (correct_output + epsilon)) * 100) < error_percentage):
            self.stop = True

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

    def train_one_input_data(self, input_data, desired_output):
        n_proc = 8
        prediction = self.predict(input_data)

        error = desired_output - prediction

        self.t += 1  # Increment the time step

        processes = []
        j = 0
        while j < len(self.weights):
            for proc in range(n_proc):
                if j >= len(self.weights):
                    break

                p = multiprocessing.Process(target=self.update_one_weight, args=(input_data, error, j,))
                processes.append(p)
                p.start()
                j += 1

            for process in processes:
                process.join()
            #self.update_one_weight(input_data, error, j)



    def train(self, input_data, desired_output, epochs, test_input, test_prediction, verbose=0, error_percentage=5):
        for epoch in range(epochs):
            if self.stop:
                break

            if verbose == 1:
                print(f"Epoch: {epoch}, prediction: {self.predict(test_input)}")

            if verbose == 2 and (epoch % 10 == 0):
                print(f"Epoch: {epoch}, prediction: {self.predict(test_input)}")


            for i in range(len(input_data)):
                if self.stop:
                    break

                self.train_one_input_data(input_data[i], desired_output[i])

            self.stop_test(test_prediction, self.predict(test_input), error_percentage)

        final_prediction = self.predict(test_input)
        print(f"Final Prediction: {final_prediction} vs {test_prediction[0]}")


def correct_output(input_data, multipliers, constant):
    desired_outputs = []
    for i in range(len(input_data)):
        desired_output = np.dot(input_data[i], multipliers) + constant
        desired_outputs.append(desired_output)

    return np.array(desired_outputs)
