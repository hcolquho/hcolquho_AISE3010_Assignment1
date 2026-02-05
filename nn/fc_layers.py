# Custom module to implement the fully connected layer from scratch
import numpy as np

# Class definition for a fully connected layer to perform Z = XW + b
class FullyConnectedLayer:

    # Initialize weights using Xavier initialisation and biases with zero
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        self.biases = np.zeros((1, output_size))

        self.X = None # To store input during forward pass
        self.Z = None # To store output during forward pass

        self.dX = None # To store gradient w.r.t input during backward pass
        self.dW = None # To store gradient w.r.t weights during backward pass

    # Forward pass
    def forward(self, input_data):

        # Store input data for backpropagation
        self.X = input_data
        # Compute the output of the layer using matrix multiplication (dot product) and adding biases
        self.Z = np.dot(input_data, self.weights) + self.biases
        return self.Z # shape is (batch_size, output_size)

    # Backward pass
    def backward(self, dZ):
        self.dW = np.dot(self.X.T, dZ)
        self.dB = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.weights.T)
        return dX