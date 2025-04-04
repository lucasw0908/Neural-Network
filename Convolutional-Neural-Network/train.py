import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

from cnn import ConvolutionalNeuralNetwork as CNN
from act import relu, drelu


learning_rate = 1e-5
data_size = (28, 28)
batch_size = 64
max_trains = 60000
epochs = 3
save = True

(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
x_trains = np.array(x_train_image).reshape(len(x_train_image), 784).astype("float64")/255
x_tests = np.array(x_test_image).reshape(len(x_test_image), 784).astype("float64")/255
y_trains = np.eye(10)[y_train_label]
y_tests = np.eye(10)[y_test_label]

nn = CNN(layers=[256, 128, 64, 10], activation_function=relu, dactivation_function=drelu, learning_rate=learning_rate, data_size=data_size, conv_layer=(1, 4))

train_loss = nn.train(x_trains, y_trains, epochs, batch_size, max_trains, save)
test_loss = nn.predict(x_tests, y_tests)


plt.plot(train_loss, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()