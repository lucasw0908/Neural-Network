import numpy as np
import matplotlib.pyplot as plt

from nn import NeuralNetwork
from act import sigmoid

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

nn = NeuralNetwork([2, 10, 2], sigmoid)
train_loss = nn.train(x_train, y_train, 10000, 1)

plt.plot(train_loss, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

while True:
    input()
    i = np.random.randint(0, 4)
    x = x_train[i]
    y = nn.forward(x)
    print(f"Input: {x}, Predicted: {y.argmax()}, Expected: {y_train[i].argmax()}")