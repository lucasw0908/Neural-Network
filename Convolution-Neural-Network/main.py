import cv2
import numpy as np
import os
from PIL import Image

from nn import NeuralNetwork
from act import relu, drelu


train_loss = []
test_loss = []
learning_rate = 1e-3
data_size = 784

nn = NeuralNetwork(layers=[784, 256, 128, 64, 10], activation_function=relu, dactivation_function=drelu, learning_rate=learning_rate)
nn.load_params()

path = os.path.join(os.path.dirname(__file__), "img")

for i in [1, 2, 3, 4, 5]:
    if not os.path.exists(f"{path}/{i}.jpg"):
        im: Image.Image = Image.open(f"{path}/{i}.png")
        im.convert('RGB').save(f"{path}/{i}.jpg","JPEG")

for i in [1, 2, 3, 4, 5]:
    img = cv2.imread(f"{path}/{i}.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow(f"image{i}", img)
    cv2.waitKey(0)
    img = cv2.resize(img, (28, 28))
    img = np.array(img).reshape(784).astype("float64")/255
    output = nn.forward(img)
    print(f"Image: {i}") 
    print(f"Prediction: {output.argmax()}")
    