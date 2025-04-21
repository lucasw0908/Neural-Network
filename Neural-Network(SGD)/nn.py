import json
import os
from datetime import datetime
from typing import Callable

import numpy as np

from np_encoder import NumpyArrayEncoder


class NeuralNetwork:
    def __init__(self, layers: list[int], activation_function: Callable, dactivation_function: Callable=None, learning_rate: float=1e-3) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.act = activation_function
        self.dact = dactivation_function or self.d(activation_function)
        self.delta = 1e-10
        self.Z: list[np.ndarray] = [np.zeros(layers[0])]
        self.W: list[np.ndarray] = [np.zeros(layers[0])]
        self.B: list[np.ndarray] = [np.zeros(layers[0])]
        self.output: list[np.ndarray]  = [np.zeros(layers[0])]

        for i in range(1, len(self.layers)):
            self.W.append(np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2/layers[i-1]))
            self.B.append(np.zeros(self.layers[i]))
            self.Z.append(np.zeros(self.layers[i]))
            self.output.append(np.zeros(self.layers[i]))
            
            
    def d(self, f: Callable) -> Callable:
        delta = 1e-10j
        def df(x): return f(x + delta).imag / delta.imag
        return df
    
            
    def softmax(self, x): 
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    

    def cross_entropy(self, y: np.ndarray) -> np.float64:
        return -np.dot(y.T, np.log(self.output[-1] + self.delta))
    

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == self.layers[0]
        self.output[0] = x

        for i in range(1, len(self.layers)):
            self.Z[i] = np.dot(self.W[i], self.output[i-1]) + self.B[i]
            if i == len(self.layers)-1: self.output[i] = self.softmax(self.Z[i])
            else: self.output[i] = self.act(self.Z[i])

        return self.output[-1]
    

    def backward(self, y: np.ndarray) -> np.ndarray:
        x = self.output[-1] - y

        for i in reversed(range(1, len(self.layers))):
            if i == len(self.layers)-1: t = x
            else: t = x * self.dact(self.Z[i])
            x = np.dot(self.W[i].T, t)
            self.W[i] -= self.learning_rate * np.outer(t, self.output[i-1])
            self.B[i] -= self.learning_rate * t
            
        return x
            

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.float64:
        self.forward(x)
        loss = self.cross_entropy(y)
        self.backward(y)
        return loss
    
    
    def predict(self, x_tests: np.ndarray, y_tests: np.ndarray) -> list[np.float64]:
        test_loss = []
        accuracy = 0
        
        for i, (x_test, y_test) in enumerate(zip(x_tests, y_tests)):
            output = self.forward(x_test)
            loss = self.cross_entropy(y_test)
            correct = output.argmax() == y_test.argmax()
            
            if correct:
                accuracy += 1
                
            test_loss.append(loss)
            print(
                "\rTest {space}{test}/{tests}, Loss: {loss}, Correct: {correct}"
                .format(
                    space=" " * (len(str(len(x_tests))) - len(str(i+1))), 
                    test=i + 1,
                    tests=len(x_tests),
                    loss='%.5f' % loss, 
                    correct=correct
                ), end=""
            )

        print(f"\r{' '*100}")
        print(f"Average test loss: {sum(test_loss) / len(test_loss)}")
        print(f"Accuracy: {accuracy / len(x_tests)}")
        
        return test_loss
    
            
    def train(self, x_trains: np.ndarray, y_trains: np.ndarray, epochs: int, batch_size: int=64, max_trains: int=60000, save: bool=False) -> list[np.float64]:
        train_loss = []
        
        for epoch in range(epochs):
            max_trains = min(max_trains, len(x_trains))
            total_loss = 0
            start_time = datetime.now()
            
            for i in range(0, max_trains, batch_size):
                x_batch = x_trains[i:i + batch_size]
                y_batch = y_trains[i:i + batch_size]
                batch_loss = 0
                
                for x_train, y_train in zip(x_batch, y_batch):
                    batch_loss += self.fit(x_train, y_train)
                    
                loss = batch_loss / batch_size
                total_loss += loss
                train_loss.append(loss)
                    
                batch = i // batch_size + 1
                batches = max_trains // batch_size + 1
                
                print(
                    "\rBatch {space}{batch}/{batchs}, Loss: {loss}, Average Loss: {avg_loss}"
                    .format(
                        space=" " * (len(str(batches)) - len(str(batch))), 
                        batch=batch, 
                        batchs=batches, 
                        loss='%.5f' % loss, 
                        avg_loss='%.5f' % (total_loss / batch)
                    ), end=""
                )
            
            print(
                "\rEpoch {space}{epoch}/{epochs}, Average Loss: {avg_loss}, Time: {time}"
                .format(
                    space=" " * (len(str(epochs)) - len(str(epoch+1))),
                    epoch=epoch + 1, 
                    epochs=epochs, 
                    avg_loss='%.5f' % (total_loss / (max_trains // batch_size + 1)), 
                    time=datetime.now() - start_time
                )
            )
            
            if save: 
                self.save_params()
            
        return train_loss
    
    
    def save_params(self, filename: str="params.json"):
        with open(os.path.join(os.path.dirname(__file__), filename), "w") as f:
            json.dump({"W": self.W, "B": self.B}, f, indent=4, cls=NumpyArrayEncoder)
            
    
    def load_params(self, filename: str="params.json"):
        with open(os.path.join(os.path.dirname(__file__), filename), "r") as f:
            params = json.load(f)
            self.W = []
            self.B = []
            for w in params["W"]: self.W.append(np.asarray(w))
            for b in params["B"]: self.B.append(np.asarray(b))