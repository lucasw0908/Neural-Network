import random

import numpy as np
from keras.datasets import mnist

from cnn import ConvolutionalNeuralNetwork as CNN
from act import relu, drelu
from image import show_image


learning_rate = 1e-5
data_size = (28, 28)

(_, _), (x_test_image, _) = mnist.load_data()
x_tests = np.array(x_test_image).reshape(len(x_test_image), 784).astype("float64")/255

nn = CNN(layers=[256, 128, 64, 10], activation_function=relu, dactivation_function=drelu, learning_rate=learning_rate, data_size=data_size, conv_layer=(1, 4))
nn.load_params()

while True:
    idx = random.randint(0, len(x_test_image))
    print(nn.forward(x_tests[idx]).argmax())
    show_image(x_tests[idx].reshape(*data_size), title=f"Image {idx} - Prediction: {nn.forward(x_tests[idx]).argmax()}")
    