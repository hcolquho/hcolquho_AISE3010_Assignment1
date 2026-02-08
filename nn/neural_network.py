# Module to implement a simple Multi-Layer Perceptron (MLP) from scratch

# nn/neural_network.py

class MLP:
    def __init__(self, layers):
        self.layers = layers # List of layers in the network (e.g., FullyConnectedLayer, ReLU, Sigmoid)

    # Forward pass through the network
    def forward(self, X):
        # Pass the input through each layer sequentially, updating X at each step
        # The output of one layer becomes the input to the next layer
        for layer in self.layers:
            X = layer.forward(X)
        return X

    # Backward pass through the network
    def backward(self, dLoss):
        # Propagate the gradient of the loss backward through each layer in reverse order
        for layer in reversed(self.layers):
            dLoss = layer.backward(dLoss) # Update dLoss for the next layer in the backward pass
        return dLoss

    # Get parameters and their corresponding gradients for all layers that have them (e.g., FullyConnectedLayer)
    def get_parameters_and_grads(self):
        params = []
        grads = []

        # Loop through each layer and check if it has weights (i.e., it's a FullyConnectedLayer)
        for layer in self.layers:
            if hasattr(layer, "weights"): # Check if the layer has weights (indicating it's a FullyConnectedLayer)
                params.append(layer.weights) # Append the weights of the layer to the params list
                params.append(layer.biases) # Append the biases of the layer to the params list
                grads.append(layer.dW) # Append the gradient w.r.t weights to the grads list
                grads.append(layer.dB) # Append the gradient w.r.t biases to the grads list

        return params, grads




