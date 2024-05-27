from typing import Callable, Iterable

import numpy as np
from layers.Layer import Layer


class Model:
    def __init__(self, layers: list[Layer]):
        """
        Initializes the model.
        """
        if len(layers) <= 0:
            raise ValueError("Model must have at least one layer")

        self.layers = layers
        self.loss = None
        self.optimizer = None
        self.input_shape = None
        self.output_shape = None

    def build(self, input_shape: tuple[int, int], loss: Callable, optimizer: Callable):
        """
        Builds the model.

        Parameters
        ----------
        input_shape : tuple[int, int]
            Input shape of the model.
        loss : Callable
            Loss function used to calculate the loss.
        optimizer : Callable
            Optimizer used to update the weights.
        """
        if not isinstance(input_shape, tuple):
            raise ValueError("Input shape must be a tuple")

        if not isinstance(loss, Callable):
            raise ValueError("Loss must be a callable")

        if not isinstance(optimizer, Callable):
            raise ValueError("Optimizer must be a callable")

        if len(input_shape) == 1:
            input_shape = (1, input_shape[0])

        self.input_shape = input_shape
        self.loss = loss
        self.optimizer = optimizer

        # Initializes the weights of the model
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                layer.build(input_shape)
                continue

            layer.build(self.layers[idx - 1].output_shape)

        # Gets the output shape of the model
        self.output_shape = self.layers[-1].output_shape

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation on the input.

        Parameters
        ----------
        input : np.ndarray
            Input to forward propagate.
        """
        y_pred = self.layers[0].forward(input)
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                continue

            y_pred = layer.forward(y_pred)

        return y_pred

    def train(self, X_train: Iterable, y_train: Iterable):
        """
        Trains the model on the passed dataset

        Parameters
        ----------
        X : Iterable
            Features of the dataset to use for training.
        y : Iterable
            Labels or Targets to use for training.
        """
        raise NotImplementedError()

    def add(self, layer: Layer):
        """
        Adds Layer to the model.

        Parameters
        ----------
        layer : Layer
            Layer to add to the model.
        """
        self.layers.append(layer)

    def evaluate(self, X_test: Iterable, y_test: Iterable):
        """
        Evaluates the model.

        Parameters
        ----------
        X_test : Iterable
            Testing dataset.
        y_test : Iterable
            True labels to use for evaluation.
        """
        raise NotImplementedError()
