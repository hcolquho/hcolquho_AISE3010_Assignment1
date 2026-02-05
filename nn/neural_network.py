# Module to implement a simple Multi-Layer Perceptron (MLP) from scratch

# nn/neural_network.py

class MLP:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dLoss):
        for layer in reversed(self.layers):
            dLoss = layer.backward(dLoss)
        return dLoss

    def get_parameters_and_grads(self):
        params = []
        grads = []

        for layer in self.layers:
            if hasattr(layer, "weights"):
                params.append(layer.weights)
                params.append(layer.biases)
                grads.append(layer.dW)
                grads.append(layer.dB)

        return params, grads




