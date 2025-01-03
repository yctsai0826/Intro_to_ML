import typing as t
import numpy as np
import torch
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        losses_of_models = []

        for model in self.learners:
            # Bootstrap sampling (with replacement)
            indices = np.random.choice(len(X_train), size=len(X_train), replace=True)

            X_sample = X_train.iloc[indices]
            y_sample = y_train[indices]

            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            X_sample = torch.FloatTensor(X_sample.values)
            y_sample = torch.FloatTensor(y_sample).unsqueeze(1)

            model_losses = []
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()

                outputs = model(X_sample)
                loss = entropy_loss(outputs, y_sample)
                loss /= len(X_sample)  # average loss over all samples

                loss.backward()
                optimizer.step()

                model_losses.append(loss.item())

            losses_of_models.append(model_losses)

        return losses_of_models

    def predict_learners(self, X) -> t.Tuple[t.Union[t.Sequence[int], t.Sequence[float]], t.Sequence[float]]:
        X = torch.FloatTensor(X.values)

        predictions = []
        probabilities = []
        for model in self.learners:
            model.eval()  # set model to evaluation mode
            with torch.no_grad():
                outputs = model(X)
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs).squeeze().tolist()
                probabilities.append(probs)
                # Convert outputs to binary predictions (0 or 1) using a threshold of 0.5
                preds = (torch.sigmoid(outputs) >= 0.5).int().squeeze().tolist()
                predictions.append(preds)

        y_pred_prob = np.array(probabilities)

        predictions = list(zip(*predictions))
        probabilities = list(zip(*probabilities))

        final_predictions = [max(set(pred), key=pred.count) for pred in predictions]

        return final_predictions, y_pred_prob

    def compute_feature_importance(self) -> t.Sequence[float]:
        feature_importances = np.zeros(self.learners[0].layer.weight.size(1))

        for model in self.learners:
            with torch.no_grad():
                weights = model.layer.weight.squeeze().numpy()
                feature_importances += np.abs(weights)

        feature_importances /= len(self.learners)

        return feature_importances.tolist()
