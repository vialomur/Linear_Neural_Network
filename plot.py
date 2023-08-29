import matplotlib.pyplot as plt


def plot_learning_process(epoch_list: list, error_list: list, bias_list: list, weight_list: list):
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(epoch_list, error_list, marker='o')
    plt.title('Error vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error')

    plt.subplot(2, 1, 2)
    plt.plot(epoch_list, bias_list, label='Bias', marker='o')
    plt.plot(epoch_list, weight_list, label='Weight', marker='o')
    plt.title('Bias and Weight vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.show()
