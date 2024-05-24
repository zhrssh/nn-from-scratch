from abc import ABC, abstractmethod
from typing import Callable, Iterable

import numpy as np


class Model(ABC):
    @abstractmethod
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
        pass

    @abstractmethod
    def forward(self, input: np.ndarray):
        """
        Performs forward propagation on the input.

        Parameters
        ----------
        input : np.ndarray
            Input to forward propagate.
        """
        pass

    @abstractmethod
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

    @abstractmethod
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
