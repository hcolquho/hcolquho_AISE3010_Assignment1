# Module to implement the Stochastic Gradient Descent with Momentum (SGDM) optimizer from scratch
import numpy as np

class SGDMOptimiser:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self, params, grads):
        for p, g in zip(params, grads):
            if id(p) not in self.velocities:
                self.velocities[id(p)] = np.zeros_like(p)
            v = self.velocities[id(p)]
            v[:] = self.momentum * v - self.lr * g
            p += v
