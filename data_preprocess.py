from PIL import Image
import numpy as np


def preprocess_image(path: str):
    # Open the JPG image
    image = Image.open(path)

    # Convert the image to grayscale
    gray_image = image.convert('L')

    # resize image
    resized_image = gray_image.resize((255, 255))

    # Convert the grayscale image to a NumPy array
    numpy_array = np.array(resized_image) / 255

    # Convert the 2D array to a 1D array using flatten()
    flattened_array = numpy_array.flatten()

    # Display the shape of the NumPy array (height, width)
    print(flattened_array.shape)

    # Optionally, display the NumPy array
    print(flattened_array)

    return flattened_array
