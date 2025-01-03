import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate=1e-4, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y, itshow=False):
        m = x.shape[0]
        n = x.shape[1]
        self.weights = np.zeros(n)    # initialized to 0
        self.intercept = 0    # initialized to 0

        for i in range(self.num_iterations):
            z = np.dot(x, self.weights) + self.intercept
            a = self.sigmod(z)    # calculate probability

            # Gradient Descent
            dw = 1/m * np.dot(x.T, (a - y))
            db = 1/m * np.sum(a - y)

            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db * 50

            if i % 1000 == 1 and itshow:
                ypred = self.predict(x)
                logger.info(f"Iteration {i}, ",
                            f"Train-Acc={accuracy_score(y, ypred):.4f}")

    def predict(self, x):
        z = np.dot(x, self.weights) + self.intercept
        prob = self.sigmod(z)
        return prob, np.where(prob >= 0.5, 1, 0)


class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None
        self.intercept = None

    def fit(self, x, y):    # x.shape = (166, 2), y.shape = (166,)
        self.m0 = np.mean(x[y == 0], axis=0)     # m.shape = (2,)
        self.m1 = np.mean(x[y == 1], axis=0)

        self.sw = (
            (x[y == 0] - self.m0).T @ (x[y == 0] - self.m0) +
            (x[y == 1] - self.m1).T @ (x[y == 1] - self.m1)
        )
        self.sb = np.outer((self.m0 - self.m1), (self.m0 - self.m1))

        sw_inv = np.linalg.inv(self.sw)
        self.w = sw_inv @ (self.m1 - self.m0)
        self.w = self.w / np.linalg.norm(self.w)    # normalize

        self.slope = self.w[1] / self.w[0]

    def predict(self, x):
        projections = np.dot(x, self.w)
        threshold = np.dot((self.m0 + self.m1) / 2, self.w)
        return (projections >= threshold).astype(int)

    def plot_projection(self, x):
        y = self.predict(x)
        self.intercept = 0

        plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], c="r")
        plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], c="b")

        max_bound = -100
        min_bound = 100

        for i in range(len(x)):
            x[i][1] -= self.intercept
            proj_point = np.dot(x[i], self.w) * self.w
            proj_point[1] += self.intercept
            x[i][1] += self.intercept
            plt.plot([x[i, 0], proj_point[0]], [x[i, 1], proj_point[1]],
                     'gray', linestyle="-", alpha=0.5)
            if y[i] == 0:
                plt.scatter(proj_point[0], proj_point[1], color='red')
            else:
                plt.scatter(proj_point[0], proj_point[1], color='blue')

            if max_bound < np.max(np.array([proj_point[0],
                                            proj_point[1], x[i, 0], x[i, 1]])):
                max_bound = np.max(np.array([proj_point[0],
                                             proj_point[1], x[i, 0], x[i, 1]]))
            if min_bound > np.min(np.array([proj_point[0],
                                            proj_point[1], x[i, 0], x[i, 1]])):
                min_bound = np.min(np.array([proj_point[0],
                                             proj_point[1], x[i, 0], x[i, 1]]))

        # Draw the projection line
        x_vals = np.array([min_bound, max_bound])
        y_vals = self.slope * x_vals + self.intercept
        plt.plot(x_vals, y_vals, 'k-')

        plt.axis('equal')
        # Title and labels
        plt.title(
            f"Projection Line: w={self.slope:.6f},\n"
            f"b={self.intercept}"
        )
        plt.show()


def compute_auc(y_trues, y_preds):
    auc = roc_auc_score(y_trues, y_preds)
    return auc


def accuracy_score(y_trues, y_preds):
    correct_predictions = np.sum(np.array(y_trues) == np.array(y_preds))
    accuracy = correct_predictions / len(y_trues)
    return accuracy


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=5e-3,
        num_iterations=15000,
    )
    LR.fit(x_train, y_train, True)
    y_pred_probs, y_pred_classes = LR.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercept: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['10', '20']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_pred_classes = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    FLD_.plot_projection(x_test)


if __name__ == '__main__':
    main()
