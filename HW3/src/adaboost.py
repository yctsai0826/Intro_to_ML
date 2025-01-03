import typing as t
import numpy as np
import torch
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 1e-4):
        # initialize the sample weights to 1/N
        self.sample_weights = np.ones(X_train.shape[0]) / X_train.shape[0]

        losses_of_models = []
        for model in self.learners:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                optimizer.zero_grad()

                outputs = model(torch.tensor(X_train.values, dtype=torch.float32)).squeeze()

                each_loss = entropy_loss(outputs, torch.tensor(y_train, dtype=torch.float32))
                weighted_loss = each_loss * torch.tensor(self.sample_weights, dtype=torch.float32)
                avg_loss = weighted_loss.mean()
                losses_of_models.append(avg_loss.item())

                avg_loss.backward()
                optimizer.step()

            with torch.no_grad():
                outputs = model(torch.tensor(X_train.values, dtype=torch.float32))
                predictions = torch.sigmoid(outputs).numpy().squeeze()

                final_pred = np.where(predictions > 0.5, 1, 0)

            incorrect = (final_pred != y_train).astype(int)
            error_rate = np.sum(incorrect * self.sample_weights) / np.sum(self.sample_weights)
            alpha = np.log((1 - error_rate) / error_rate + 1e-10) / 2
            self.alphas.append(alpha)
            self.sample_weights = self.sample_weights * np.exp(-1 * alpha * (2 * incorrect - 1))
            self.sample_weights /= np.sum(self.sample_weights)

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        X = X.values    # Transform to numpy array
        y_preds = np.zeros(X.shape[0])
        y_pred_probs = []

        for model, alpha in zip(self.learners, self.alphas):
            X_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = model(X_tensor)
            predictions = torch.sigmoid(outputs).detach().numpy().squeeze()
            y_preds += alpha * (2 * (predictions > 0.5) - 1)    # Convert to 1 and -1
            y_pred_probs.append(predictions)

        y_pred_classes = (y_preds > 0).astype(int)  # Convert to 1 and 0

        return y_pred_classes, y_pred_probs

    def compute_feature_importance(self) -> t.Sequence[float]:
        feature_importance = np.zeros(self.learners[0].layer.in_features)
        for alpha, model in zip(self.alphas, self.learners):
            with torch.no_grad():
                importance = np.abs(model.layer.weight.numpy()).squeeze()
            feature_importance += alpha * importance
        return feature_importance.tolist()
