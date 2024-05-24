import numpy as np

from layers.Dense import Dense


class TestDense:
    def activation(self, X):
        return X

    def test_dense_init(self):
        dense = Dense(units=3, activation=self.activation)
        dense.build(input_shape=(2,))

        assert dense.units == 3
        assert dense.activation == self.activation
        assert dense._weights.shape == (2, 3)
        assert dense._bias.shape == (1, 3)

    def test_dense_forward(self):
        dense = Dense(units=3, activation=self.activation)
        dense.build(input_shape=(10,))

        assert dense._weights.shape == (10, 3)
        assert dense._bias.shape == (1, 3)

        X = np.random.rand(10)
        y_pred = dense.forward(X.T)

        assert y_pred.shape == (1, 3)
        assert y_pred.dtype == np.float64

        print("Predictions:", y_pred)
