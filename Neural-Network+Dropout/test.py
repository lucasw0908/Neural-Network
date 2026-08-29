import random

import matplotlib.pyplot as plt
import numpy as np
from act import drelu, relu
from keras.datasets import mnist
from nn import NeuralNetwork

learning_rate = 1e-3

(_, _), (x_test_image, _) = mnist.load_data()
x_tests = np.array(x_test_image).reshape(len(x_test_image), 784).astype("float64") / 255

nn = NeuralNetwork(
    layers=[784, 256, 128, 64, 10], activation_function=relu, dactivation_function=drelu, learning_rate=learning_rate
)
nn.load_params()

while True:
    idx = random.randint(0, len(x_test_image))
    print(nn.forward(x_tests[idx]).argmax())
    plt.imshow(x_test_image[idx], interpolation='nearest')
    plt.show()
