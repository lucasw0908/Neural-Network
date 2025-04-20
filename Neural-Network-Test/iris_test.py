import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from nn import NeuralNetwork
from act import relu, drelu


iris = load_iris()
x_data = (iris.data - np.mean(iris.data, axis=0)) / np.std(iris.data, axis=0)
y_data = np.eye(3)[iris.target]

x_trains, x_tests, y_trains, y_tests = train_test_split(
    x_data, y_data, test_size=0.2, stratify=iris.target, random_state=42
)

nn = NeuralNetwork([4, 8, 6, 3], relu, drelu)
train_loss = nn.train(x_trains, y_trains, 500)
nn.predict(x_tests, y_tests)

plt.plot(train_loss, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

while True:
    input()
    i = np.random.randint(0, 150)
    x = x_data[i]
    y = nn.forward(x)
    print(f"Input: {x}, Predicted: {y.argmax()}, Expected: {y_data[i].argmax()}")
    