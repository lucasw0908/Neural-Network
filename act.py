import numpy as np

def relu(x): 
    return np.maximum(0, x)

def drelu(x):
    return np.where(x > 0, 1, 0)