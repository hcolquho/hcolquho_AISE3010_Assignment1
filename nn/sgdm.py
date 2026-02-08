# Module to implement the Stochastic Gradient Descent with Momentum (SGDM) optimizer from scratch
import numpy as np

class SGDMOptimiser:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr # Learning rate
        self.momentum = momentum # momentum factor
        self.velocities = {} # To store velocities for each parameter

    # Update parameters using SGDM
    def step(self, params, grads):
        for p, g in zip(params, grads):
            if id(p) not in self.velocities:
                self.velocities[id(p)] = np.zeros_like(p) # Initialize velocity for this parameter if not already done
            v = self.velocities[id(p)] # Get the velocity for this parameter from the dictionary
            v[:] = self.momentum * v - self.lr * g  # Update velocity using momentum and current gradient
            p += v # Update parameter using the velocity (which incorporates momentum)
