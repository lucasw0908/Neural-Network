import random
import json
from datetime import datetime
from typing import Callable

import numpy as np

from np_encoder import NumpyArrayEncoder


class NeuralNetwork:
    def __init__(self, layers: list[int], activation_function: Callable, dactivation_function: Callable=None, learning_rate: float=1e-3, dropout: float=0.5) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.act = activation_function
        self.dact = dactivation_function or self.d(activation_function)
        self.dropout = dropout
        self.is_training = True
        self.delta = 1e-10
        self.Z: list[np.ndarray] = [np.zeros(layers[0])]
        self.W: list[np.ndarray] = [np.zeros(layers[0])]
        self.B: list[np.ndarray] = [np.zeros(layers[0])]
        self.output: list[np.ndarray]  = [np.zeros(layers[0])]
        
        self.m: list[np.ndarray] = [0 for _ in range((len(layers)-1)*2)]
        self.v: list[np.ndarray] = [0 for _ in range((len(layers)-1)*2)]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.gd_times = 1
        self.gd_tag = 0

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
    
    
    def adam(self, grad: np.ndarray, t: int) -> np.ndarray:

        self.m[self.gd_tag] = self.beta1 * self.m[self.gd_tag] + (1 - self.beta1) * grad
        self.v[self.gd_tag] = self.beta2 * self.v[self.gd_tag] + (1 - self.beta2) * np.square(grad)
        
        m_hat = self.m[self.gd_tag] / (1 - np.float_power(self.beta1, t))
        v_hat = self.v[self.gd_tag] / (1 - np.float_power(self.beta2, t))
        
        self.gd_tag += 1
        
        return self.learning_rate * m_hat / (np.sqrt(v_hat) + self.delta)
    

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == self.layers[0]
        self.output[0] = x

        for i in range(1, len(self.layers)):
            self.Z[i] = (np.dot(self.W[i], self.output[i-1]) + self.B[i])
            
            if self.is_training: 
                self.Z[i] = np.where(np.random.rand(*self.Z[i].shape) < self.dropout, self.Z[i], 0) / (1 - self.dropout)
                
            if i == len(self.layers)-1: self.output[i] = self.softmax(self.Z[i])
            else: self.output[i] = self.act(self.Z[i])

        return self.output[-1]
    

    def backward(self, y: np.ndarray) -> np.ndarray:
        x = self.output[-1] - y
        
        for i in reversed(range(1, len(self.layers))):
            t = x * self.dact(self.Z[i])
            x = np.dot(self.W[i].T, t)
            self.W[i] -= self.adam(np.outer(t, self.output[i-1]), self.gd_times)
            self.B[i] -= self.adam(t, self.gd_times)
            
        self.gd_times += 1
        self.gd_tag = 0
        
        return x
            

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.float64:
        self.forward(x)
        loss = self.cross_entropy(y)
        self.backward(y)
        return loss
    
    
    def predict(self, x_tests: np.ndarray, y_tests: np.ndarray) -> list[np.float64]:
        self.is_training = False
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
                "Test {space}{test}/{tests}, Loss: {loss}, Correct: {correct}"
                .format(
                    space=" " * (len(str(len(x_tests))) - len(str(i+1))), 
                    test=i + 1,
                    tests=len(x_tests),
                    loss='%.5f' % loss, 
                    correct=correct
                )
            )
            
        print(f"Average test loss: {sum(test_loss) / len(test_loss)}")
        print(f"Accuracy: {accuracy / len(x_tests)}")
        
        return test_loss
    
            
    def train(self, x_trains: np.ndarray, y_trains: np.ndarray, epochs: int, batch_size: int=64, max_trains: int=60000, save: bool=False) -> list[np.float64]:
        self.is_training = True
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
                batchs = max_trains // batch_size + 1
                
                print(
                    "Batch {space}{batch}/{batchs}, Loss: {loss}, Average Loss: {avg_loss}"
                    .format(
                        space=" " * (len(str(batchs)) - len(str(batch))), 
                        batch=batch, 
                        batchs=batchs, 
                        loss='%.5f' % loss, 
                        avg_loss='%.5f' % (total_loss / batch)
                    )
                )
            
            print(
                "Epoch {space}{epoch}/{epochs}, Loss: {loss}, Save: {save}, Time: {time}"
                .format(
                    space=" " * (len(str(epochs)) + len(str(batchs)) * 2 - len(str(epoch)) * 3), 
                    epoch=epoch + 1, 
                    epochs=epochs, 
                    loss='%.5f' % (total_loss / (max_trains // batch_size + 1)), 
                    save=save, 
                    time=datetime.now() - start_time
                )
            )
            
            if save: 
                self.save_params()
            
        return train_loss
    
    
    def save_params(self, filename: str="params.json"):
        with open(filename, "w") as f:
            json.dump({"W": self.W, "B": self.B}, f, indent=4, cls=NumpyArrayEncoder)
            
    
    def load_params(self, filename: str="params.json"):
        with open(filename, "r") as f:
            params = json.load(f)
            self.W = []
            self.B = []
            for w in params["W"]: self.W.append(np.asarray(w))
            for b in params["B"]: self.B.append(np.asarray(b))