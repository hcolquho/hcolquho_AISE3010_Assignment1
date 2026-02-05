# Module to implement MSE and Cross-Entropy loss functions from scratch
import numpy as np

# Mean Squared Error (MSE) Loss
class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        n = self.y_true.shape[0]
        return (2 / n) * (self.y_pred - self.y_true)
