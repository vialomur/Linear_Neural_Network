import numpy as np
from plot import plot_learning_process

# Define the model parameters
weight = 0.5
bias = 0.2
learning_rate = 0.01
epochs = 8

# Training data
input_data = np.array([1.0, 2.0, 3.0, 4.0])
desired_output = np.array([num * 3 + 3 for num in input_data])  # 3x + 3

epoch_list = []
error_list = []
bias_list = []
weight_list = []
test_input = 5.0
test_prediction = 3 * test_input + 3
prediction = 0
# Training loop
for epoch in range(epochs):
    epoch_list.append(epoch)
    total_error = 0

    print(f"Epoch: {epoch}, prediction: {test_input * weight + bias}")
    for i in range(len(input_data)):
        # Make prediction
        prediction = input_data[i] * weight + bias

        # Calculate error
        error = desired_output[i] - prediction

        # Update weights and bias using gradient descent
        weight += learning_rate * error * input_data[i]
        bias += learning_rate * error

    total_error = abs(test_prediction - prediction)
    # Store values for plotting
    error_list.append(total_error)
    bias_list.append(bias)
    weight_list.append(weight)

# Test the trained model
final_prediction = test_input * weight + bias
plot_learning_process(epoch_list, error_list, bias_list, weight_list)
print("Final Prediction:", final_prediction)
