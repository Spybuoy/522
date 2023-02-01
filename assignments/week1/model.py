import numpy as np


class LinearRegression:
    """
    LinearRegression Class
    """

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        fit
        """
        # stacking ones to get biases
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Closeform to get weights
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y

        # since we've added bias, we need to separate them
        self.b = self.w[0]
        self.w = self.w[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict
        """
        # stacking ones to get biases
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # predicting using wx+b
        y_pred = X @ np.hstack((self.b, self.w))
        return y_pred


class GradientDescentLinearRegression(LinearRegression):
    """
    GradientDescentLinearRegression
    """

    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        fit
        """
        # stacking ones to get biases
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Setting initial weights
        self.w = np.zeros(X.shape[1])
        self.b = 0

        # loop for gradient descent
        for i in range(epochs):
            # predicting using wx+b
            y_pred = X @ self.w + self.b

            # gradient calc, we just are substituting to already differentiated equation
            dw = (2 / X.shape[0]) * X.T @ (y_pred - y)
            db = (2 / X.shape[0]) * np.sum(y_pred - y)

            # updating the weights
            self.w -= lr * dw
            self.b -= lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict
        """
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # predicting using wx+b
        y_pred = X @ self.w + self.b
        return y_pred
