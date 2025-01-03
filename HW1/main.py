import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from loguru import logger


class LinearRegressionCloseform:
    def fit(self, x, y):
        ones = np.ones((x.shape[0], 1))
        x_p = np.hstack((x, ones))

        # formula: w = (X^T * X)^-1 * X^T * y
        weights = np.linalg.inv(x_p.T @ x_p) @ x_p.T @ y
        self.weights = weights[:-1]
        self.intercept = weights[-1]

    def predict(self, x):
        return x @ self.weights + self.intercept

    def compute_mse(self, prediction, ground_truth):
        self.mse = np.mean((prediction - ground_truth) ** 2)
        return self.mse


def validate_with_sklearn(train_x, train_y):
    model = LinearRegression()
    model.fit(train_x, train_y)
    logger.info(
        f'scikit-learn weights: {model.coef_}, '
        f'scikit-learn intercept: {model.intercept_}'
    )


class LinearRegressionGradientdescent:
    def fit(self, x, y, learning_rate=1e-4, epochs=10000, batch_size=50):
        #   initialize weights and intercept
        self.weights = np.zeros((x.shape[1], 1))
        self.intercept = 0

        losses = []

        for epoch in range(epochs):
            y_pred = x @ self.weights + self.intercept
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)

            #   calculate gradients
            dw = (x.T @ (y_pred - y)) / x.shape[0]
            db = np.mean(y_pred - y)

            #   update weights and intercept
            self.weights -= learning_rate * dw
            self.intercept -= learning_rate * db * 50

            if epoch % 10000 == 0:
                logger.info(f'EPOCH {epoch}, loss={loss}')

        return losses

    def predict(self, x):
        return x @ self.weights + self.intercept

    def compute_mse(self, prediction, ground_truth):
        self.mse = np.mean((prediction - ground_truth) ** 2)
        return self.mse


def plot_curve(losses):
    epochs = range(1, len(losses) + 1)  # x-axis: epoch numbers

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label='Training Loss', color='blue')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    print("Start loading data...")
    train_data = pd.read_csv("./train.csv")
    test_data = pd.read_csv("./test.csv")

    x_train = train_data.drop(["Performance Index"], axis=1).to_numpy()
    y_train = train_data["Performance Index"].to_numpy().reshape(-1, 1)
    x_test = test_data.drop(["Performance Index"], axis=1).to_numpy()
    y_test = test_data["Performance Index"].to_numpy().reshape(-1, 1)

    #   Part 1-1. Linear regression model-Close form
    print("Part 1-1. Linear regression model-Close")
    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(x_train, y_train)
    logger.info(
        f'LR_CF.weights={LR_CF.weights.reshape(1, -1)}, '
        f'LR_CF.intercept={LR_CF.intercept}'
    )
    y_pred = LR_CF.predict(x_test)
    LR_CF_MSE = LR_CF.compute_mse(y_pred, y_test)

    #   Part 1-1. Validate weights with sklearn
    validate_with_sklearn(x_train, y_train)

    #   Part 1-2. Linear regression model-Gradient descent
    print("Part 1-2. Linear regression model-Gradient descent")
    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(x_train, y_train, learning_rate=1e-4, epochs=100000)
    plot_curve(losses)
    logger.info(
        f'LR_GD.weights: {LR_GD.weights.reshape(1, -1)}, '
        f'LR_GD.intercept: {LR_GD.intercept}'
    )
    y_pred = LR_GD.predict(x_test)
    LR_GD_MSE = LR_GD.compute_mse(y_pred, y_test)

    #   Part 1-2. Compare MSE between close form and gradient descent
    logger.info(f'Prediction difference: {np.abs(LR_CF_MSE - LR_GD_MSE):.4f}')
    logger.info(
        f'mse_cf={LR_CF_MSE:.4f}, mse_gd={LR_GD_MSE:.4f}. '
        f'Difference: {(np.abs(LR_CF_MSE - LR_GD_MSE) / LR_CF_MSE) * 100:.4f}%'
    )


if __name__ == "__main__":
    main()
