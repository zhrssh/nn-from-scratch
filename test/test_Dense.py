import numpy as np

from activations.ReLU import ReLU
from activations.sigmoid import sigmoid
from layers.Dense import Dense
from models.Model import Model


class TestDense:
    def loss(self, X):
        return X

    def optimizer(self, X):
        return X

    def test_dense_init(self):
        dense = Dense(units=3, activation=sigmoid)
        dense.build(input_shape=(2,))

        assert dense.units == 3
        assert dense.activation == sigmoid
        assert dense._weights.shape == (2, 3)
        assert dense._biases.shape == (1, 3)

    def test_dense_forward(self):
        dense = Dense(units=3, activation=sigmoid)
        dense.build(input_shape=(10,))

        assert dense._weights.shape == (10, 3)
        assert dense._biases.shape == (1, 3)

        X = np.random.rand(10)
        y_pred = dense.forward(X.T)

        assert y_pred.shape == (1, 3)
        assert y_pred.dtype == np.float64

        print("Predictions:", y_pred)

    def test_dense_multiple(self):
        test_input = np.random.rand(1, 10)
        print("Test input:", test_input)

        layers = [
            Dense(units=3, activation=ReLU),
            Dense(units=3, activation=ReLU),
            Dense(units=5, activation=sigmoid),
        ]

        model = Model(layers=layers)
        model.build(input_shape=(10,), loss=self.loss, optimizer=self.optimizer)

        assert isinstance(model, Model)
        assert model.input_shape == (1, 10)
        assert model.output_shape == (1, 5)

        y_pred = model.forward(test_input)
        print("Model predictions:", y_pred)

        assert y_pred.shape == (1, 5)
