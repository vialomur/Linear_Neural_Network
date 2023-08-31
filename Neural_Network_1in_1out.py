from matplotlib import pyplot as plt


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