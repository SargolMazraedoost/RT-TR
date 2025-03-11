#!/usr/bin/env python
"""
@File: metrics.py
@Author: Sargol Mazraedoost
@Created: 2025/02/14
@Contact: sargol@pukyong.ac.kr
"""

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
)


def compute_metrics(eval_pred, scaler=None):
    """
    Compute the metrics for the given predictions and labels.

    Args:
        eval_pred (Tuple[torch.Tensor, torch.Tensor]): Tuple containing the predictions and labels.
        scaler (sklearn.preprocessing.TransformerMixin, optional): The scaler used to transform the data. Defaults to None.

    Returns:
        Dict[str, str]: Dictionary containing the metrics.
    """
    predictions, labels = eval_pred
    # Ensure predictions and labels are 1D arrays
    predictions = predictions.squeeze()
    labels = labels.squeeze()

    mae_loss = mean_absolute_error(labels, predictions)

    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    labels = scaler.inverse_transform(labels.reshape(-1, 1)).flatten()

    # Compute metrics
    mae = mean_absolute_error(labels, predictions)

    # Avoid division by zero in MAPE calculation
    mape = mean_absolute_percentage_error(labels, predictions)

    r2 = r2_score(labels, predictions)
    medae = median_absolute_error(labels, predictions)

    return {
        "y-y^": f"{mae_loss:.2f}",
        "mae": f"{mae:.4f}",
        "mape": f"{mape * 100:.2f}%",
        "r2": f"{r2:.4f}",
        "medae": f"{medae:.4f}",
    }
