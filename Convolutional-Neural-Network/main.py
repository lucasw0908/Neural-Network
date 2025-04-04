import cv2
import os
import numpy as np

from cnn import ConvolutionalNeuralNetwork as CNN
from act import relu, drelu
from image import show_image


learning_rate = 1e-5
data_size = (28, 28)
img_count = 10

nn = CNN(layers=[256, 128, 64, 10], activation_function=relu, dactivation_function=drelu, learning_rate=learning_rate, data_size=data_size, conv_layer=(1, 4))
nn.load_params()

path = os.path.join(os.path.dirname(__file__), "img")

for i in range(1, img_count+1):
    img = cv2.imread(f"{path}/{i}.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, data_size, interpolation=cv2.INTER_AREA)
    img = np.array(img).reshape(784).astype("float64")/255
    for j in range(len(img)): img[j] = 1 - img[j]
    output = nn.forward(img)
    print(f"Image {i} - Prediction: {output.argmax()}")
    show_image(img.reshape(data_size), title=f"Image {i} - Prediction: {output.argmax()}")
    