import numpy as np
import json
from typing import Callable

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
            self.W.append(np.random.randn(self.layers[i], self.layers[i-1]) / np.sqrt(layers[i-1]))
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
    

    def backward(self, y: np.ndarray) -> None:
        x = self.output[-1] - y

        for i in range(len(self.layers)-1, 0, -1):
            t = x * self.dact(self.Z[i])
            x = np.dot(self.W[i].T, t)
            self.W[i] -= self.learning_rate * np.outer(t, self.output[i-1])
            self.B[i] -= self.learning_rate * t
            

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
            print(f"Test Data: {i+1}/{len(x_tests)}, Loss: {loss}, Correct: {correct}")
            
        print(f"Average test loss: {sum(test_loss) / len(test_loss)}")
        print(f"Accuracy: {accuracy / len(x_tests)}")
        
        return test_loss
    
            
    def train(self, x_trains: np.ndarray, y_trains: np.ndarray, epochs: int, batch_size: int=64, max_trains: int=60000, save: bool=False) -> list[np.float64]:
        train_loss = []
        
        for epoch in range(epochs):
            max_trains = min(max_trains, len(x_trains))
            batch_loss = 0
            
            for i in range(0, max_trains, batch_size):
                x_batch = x_trains[i:i + batch_size]
                y_batch = y_trains[i:i + batch_size]
                
                for x_train, y_train in zip(x_batch, y_batch):
                    batch_loss += self.fit(x_train, y_train)
                    
                print(f"Batch {i//batch_size+1}/{max_trains//batch_size+1}, Loss: {batch_loss/(i+1)}")
                
            avg_loss = batch_loss / max_trains
            train_loss.append(avg_loss)
            
            if save:
                self.save_params()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Save: {save}")
            
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