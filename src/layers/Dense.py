from typing import Callable

import numpy as np

from .Layer import Layer


class Dense(Layer):
    def __init__(self, units: int, activation: Callable):
        super().__init__()
        self.units = units
        self.activation = activation
        self.output_shape = (1, self.units)

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)

    def build(self, input_shape: tuple[int, int]):
        if len(input_shape) == 1:
            input_shape = (1, input_shape[0])

        self._weights = np.random.rand(input_shape[1], self.units)
        self._biases = np.random.rand(1, self.units)

    def forward(self, input: np.ndarray):
        X = np.matmul(input, self._weights) + self._biases
        X = self.activation(X)
        return X

    def update(self, new_weights: np.ndarray, new_biases: np.ndarray):
        self._weights = new_weights
        self._biases = new_biases
