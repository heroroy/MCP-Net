"""
eval.py

Evaluates a trained MCPNet model on test data.
Outputs metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and saves predictions.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from models import MCPNet
from dataset import create_generators


def evaluate(model_path="results/best_model.h5", data_root="data/Arthritis", batch_size=32):
    """
    Evaluates MCPNet on test dataset.
    Args:
        model_path: Path to saved Keras model checkpoint.
        data_root: Path to dataset.
        batch_size: Batch size.
    """
    _, _, test_gen = create_generators(data_root, batch_size=batch_size)

    model = MCPNet(input_shape=(224, 224, 3))
    model.load_weights(model_path)

    preds = model.predict(test_gen)
    y_pred = (preds > 0.5).astype(int)
    y_true = test_gen.classes

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, preds)
    }

    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    np.savetxt("results/predictions.csv", np.column_stack([y_true, preds]), delimiter=",",
               header="True,PredictedProb", comments="")


if __name__ == "__main__":
    evaluate()
