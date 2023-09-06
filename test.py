import numpy as np

class InputLayer:
    def __init__(self, input_data, learning_rate):
        np.random.seed(0)  # Set seed for reproducibility
        self.weights = np.random.rand(len(input_data))
        self.bias = np.random.rand()
        self.values = input_data
        self.learning_rate = learning_rate

    def predict(self):
        return np.dot(self.values, self.weights) + self.bias

    def update_one_weight(self, input_data, error, j):
        gradient = error * input_data[j]
        self.weights[j] += self.learning_rate * gradient

    def train_one_input_data(self, desired_output):
        prediction = self.predict()
        error = desired_output - prediction

        for j in range(len(self.weights)):
            self.update_one_weight(self.values, error, j)


class NeuralNetwork3:
    def __init__(self, input_data: np.array, learning_rate: float, output_size: int):
        self.input_layer = InputLayer(input_data, learning_rate)
        self.learning_rate = learning_rate
        self.output_weights = np.random.rand(output_size)
        self.stop = False

    def stop_test(self, correct_output, prediction, error_percentage):
        epsilon = 1e-9
        error_percent = np.abs((correct_output - prediction) / (correct_output + epsilon)) * 100
        if np.all(error_percent < error_percentage):
            self.stop = True

    def predict(self, input_data):
        self.input_layer.values = input_data
        return self.input_layer.predict()

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

                self.input_layer.values = input_data[i]
                self.input_layer.train_one_input_data(desired_output[i])

            self.stop_test(test_prediction, self.predict(test_input), error_percentage)

        final_prediction = self.predict(test_input)
        print(f"Final Prediction: {final_prediction} vs {test_prediction}")


def correct_output(input_data, multipliers, constant):
    desired_outputs = []
    for i in range(len(input_data)):
        desired_output = np.dot(input_data[i], multipliers) + constant
        desired_outputs.append(desired_output)

    return np.array(desired_outputs)

# Example usage:
input_data = np.array([[1, 2], [2, 3], [3, 4]])
multipliers = np.array([2, 3])
constant = 1
desired_output = correct_output(input_data, multipliers, constant)
test_input = np.array([4, 5])
test_prediction = np.dot(test_input, multipliers) + constant

nn = NeuralNetwork3(input_data[0], learning_rate=0.01, output_size=1)
nn.train(input_data, desired_output, epochs=1000, test_input=test_input, test_prediction=test_prediction, verbose=1)
