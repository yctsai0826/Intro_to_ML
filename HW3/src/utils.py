import typing as t
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


def preprocess(df: pd.DataFrame):
    num_features = df.select_dtypes(include=[np.number]).columns
    df[num_features] = df[num_features].fillna(df[num_features].mean())
    df[num_features] = (df[num_features] - df[num_features].mean()) / df[num_features].std()

    object_features = df.select_dtypes(include=['object']).columns
    df[object_features] = df[object_features].fillna(df[object_features].mode())
    for col in object_features:
        df[col] = df[col].astype('category').cat.codes

    return df


class WeakClassifier(nn.Module):
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        self.layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.layer(x)


def accuracy_score(y_trues, y_preds) -> float:
    raise NotImplementedError


def entropy_loss(outputs, targets):
    probs = torch.sigmoid(outputs)
    probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
    loss = -torch.mean(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
    return loss


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    plt.figure()

    # Ensure y_trues is array-like
    y_trues = np.array(y_trues)

    for i, preds in enumerate(y_preds):
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_trues, preds)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for each learner
        plt.plot(fpr, tpr, lw=1, label=f'Learner {i+1} (AUC = {roc_auc:.2f})')

    # Plot random baseline
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Learner')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(fpath)
    plt.close()


def plot_feature_importance(feature_names, importance_values, title="Feature Importance", fpath=None):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance_values, color='steelblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature Name')
    plt.title(title)
    plt.tight_layout()  # Adjust layout to fit labels properly

    # Save the plot if a file path is provided
    if fpath:
        plt.savefig(fpath)
    else:
        plt.show()
