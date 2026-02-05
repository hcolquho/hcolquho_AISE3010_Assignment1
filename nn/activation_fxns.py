# Module to implement activation functions from scratch
import numpy as np

# ReLu
class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dZ):
        dX = dZ.copy()
        dX[self.X <= 0] = 0
        return dX

    
# Sigmoid
class Sigmoid:

    # Forward pass
    def forward(self, X):
        self.X = X
        return 1 / (1 + np.exp(-X)) # sigmoid function from sratch
    
    # Backward pass
    def backward(self, dZ):
        dX = dZ.copy()
        s = self.forward(self.X) # compute sigmoid of X
        return dX * s * (1 - s) # element-wise multiplication