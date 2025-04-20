import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from nn import NeuralNetwork


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

data_size = 150
iris = load_iris()

indices = np.arange(data_size)
x_trains = (lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0))(np.array(iris.data, dtype=np.float64))
y_trains = np.array(np.eye(3)[iris.target], dtype=np.float64)

nn = NeuralNetwork([4, 5, 3], sigmoid)
train_loss = nn.train(x_trains[indices], y_trains[indices], epochs=1000)

plt.plot(train_loss, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

while True:
    input()
    i = np.random.randint(0, data_size)
    x = x_trains[i]
    y = nn.forward(x)
    print(f"Input: {x}, Predicted: {y.argmax()}, Expected: {y_trains[i].argmax()}")
    