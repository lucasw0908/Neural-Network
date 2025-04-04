import json
from typing import Callable

import numpy as np

from nn import NeuralNetwork
from np_encoder import NumpyArrayEncoder
from image import show_image

class ConvolutionalNeuralNetwork(NeuralNetwork):
    def __init__(
            self,
            layers: list[int], conv_layer: tuple[int, int],
            activation_function: Callable, dactivation_function: Callable=None, 
            learning_rate: float=1e-3, data_size: tuple[int, int]=(28, 28),
            
            conv_kernel_size: tuple[int, int]=(5, 5), conv_strides: tuple[int, int]=(1, 1),
            pool_kernel_size: tuple[int, int]=(2, 2), pool_strides: tuple[int, int]=(2, 2),
        ) -> None:
        
        self.conv_layer = conv_layer # 0: conv layer, 1: input layer image count
        self.data_size = data_size
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.pool_kernel_size = pool_kernel_size
        self.pool_strides = pool_strides
        
        self.conv_data_size = self.calc_conv_data_size()
        layers.insert(0, self.conv_data_size)
        
        super().__init__(layers, activation_function, dactivation_function, learning_rate)
        
        self.kernels = [
            [
                np.random.normal(
                    loc=0, 
                    scale=np.sqrt(2.0/(((c_in:=1) * (c_out:=conv_layer[0]*conv_layer[1])))), 
                    size=(conv_kernel_size[0], conv_kernel_size[1])
                ) 
                for _ in range(conv_layer[0])
            ] 
            for _ in range(conv_layer[1])
        ]
        
        self.X: list[list] = []
        self.pooling_maximum: list[list[list[tuple]]] = []
        
        print(f"Conv data size: {self.conv_data_size}")
        
    
    def calc_conv_data_size(self) -> int:
        
        def calc(ds: tuple[int, int]) -> int:
            conv_fp = [(ds[i]-self.conv_kernel_size[i])/(self.conv_strides[i])+1 for i in [0, 1]]
            pool_fp = [(conv_fp[i]-self.pool_kernel_size[i])/(self.pool_strides[i])+1 for i in [0, 1]]
            return (pool_fp[0], pool_fp[1])
        
        ds = self.data_size
        for _ in range(self.conv_layer[0]): ds = calc(ds)

        return int(ds[0] * ds[1] * self.conv_layer[1])
        
        
    def conv(self, x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        c, cx, cy = self.conv_kernel_size, self.conv_kernel_size[0], self.conv_kernel_size[1]
        
        output_shape = [int((x.shape[i]-c[i])/(self.conv_strides[i])+1) for i in [0, 1]]
        output = np.zeros(output_shape)
        
        for i, ix in enumerate(range(0, x.shape[0]-cx+1, self.conv_strides[0])):
            for j, jx in enumerate(range(0, x.shape[1]-cy+1, self.conv_strides[1])):
                output[i, j] = np.sum(x[ix:ix+cx, jx:jx+cy] * kernel)

        return output
    
    
    def dconv(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        cx, cy = self.conv_kernel_size[0], self.conv_kernel_size[1]
        d_kernel = np.zeros(self.conv_kernel_size)

        for i, ix in enumerate(range(0, x.shape[0]-cx+1, self.conv_strides[0])):
            for j, jx in enumerate(range(0, x.shape[1]-cy+1, self.conv_strides[1])):
                d_kernel += x[ix:ix+cx, jx:jx+cy] * dy[i, j]
                
        return d_kernel
    
    
    def max_pooling(self, x: np.ndarray, pooling_maximum: list[list[tuple]]) -> np.ndarray:
        p, px, py = self.pool_kernel_size, self.pool_kernel_size[0], self.pool_kernel_size[1]
        
        output_shape = [int((x.shape[i]-p[i])/(self.pool_strides[i])+1) for i in [0, 1]]
        output = np.zeros(output_shape)
        
        for i, ix in enumerate(range(0, x.shape[0]-px+1, self.pool_strides[0])):
            pooling_maximum.append([])
            for j, jx in enumerate(range(0, x.shape[1]-px+1, self.pool_strides[1])):
                t = x[ix:ix+px, jx:jx+py]
                output[i, j] = np.max(t)
                pooling_maximum[i].append([])
                pooling_maximum[i][j].append(t.argmax())
                
        return output
    
    
    def dmax_pooling(self, x: np.ndarray, pooling_maximum: list[list[tuple]]) -> np.ndarray:
        p, px, py = self.pool_kernel_size, self.pool_kernel_size[0], self.pool_kernel_size[1]
        output_shape = [int(((x.shape[i]-1)*self.pool_strides[i])+p[i]) for i in [0, 1]]
        output = np.zeros(output_shape)
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                argmax = pooling_maximum[i][j][0]
                output[i*px+argmax//px, j*py+argmax%px] = x[i, j]
                
        return output
    

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[0] == self.data_size[0] * self.data_size[1]
        x = x.reshape(self.data_size[0], self.data_size[1])
        x_conv = []
        self.pooling_maximum = []
        self.X = []
        
        for i in range(len(self.kernels)):
            self.pooling_maximum.append([])
            _x = x.copy()
            self.X.append([])
            for j in range(len(self.kernels[i])):
                self.pooling_maximum[i].append([])
                self.X[i].append(_x.copy())
                _x = self.conv(_x, self.kernels[i][j])
                _x = self.max_pooling(_x, self.pooling_maximum[i][j])
            
            x_conv.extend(_x.reshape(-1))
            
        x_conv = np.array(x_conv).astype("float64")
        assert x_conv.shape[0] == self.layers[0]
        return super().forward(x_conv)
    
    
    def backward(self, y: np.ndarray) -> None:
        x_conv = super().backward(y)
        size = self.conv_data_size / self.conv_layer[1]
        x_conv = x_conv.reshape(self.conv_layer[1], int(np.sqrt(size)), int(np.sqrt(size)))
        
        for i in range(len(self.kernels)-1, -1, -1):
            x = x_conv[i].copy()
            for j in range(len(self.kernels[i])-1, -1, -1):
                x = self.dmax_pooling(x, self.pooling_maximum[i][j])
                t = self.dconv(self.X[i][j], x)
                self.kernels[i][j] -= self.learning_rate * t
        
        
    def save_params(self, filename = "cnn_params.json"):
        super().save_params(filename)
        
        with open(filename, "r") as f: params = json.load(f)
        params["kernels"] = self.kernels
        with open(filename, "w") as f: json.dump(params, f, cls=NumpyArrayEncoder)
        
        
    def load_params(self, filename = "cnn_params.json"):
        super().load_params(filename)
        
        with open(filename, "r") as f: params = json.load(f)
        self.kernels = params["kernels"]