from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    @abstractmethod
    def build(self, input_shape: tuple[int, int]):
        """
        Builds and initializes the weights of the layer based on the input shape

        Parameters
        ----------
        input_shape : _type_
            _description_
        output_shape : _type_
            _description_
        """
        pass

    @abstractmethod
    def forward(self, input: np.ndarray):
        """
        Performs a forward pass on the input array.

        Parameters
        ----------
        input : np.ndarray
            Numpy array input to forward pass.
        """
        pass

    @abstractmethod
    def update(self, new_weights: np.ndarray, new_biases: np.ndarray):
        """
        Updates the weights of the layer.

        Parameters
        ----------
        new_weights : np.ndarray
            New weights to set on the layer.
        """
        pass
