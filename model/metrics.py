"""
Metrics calculation functions
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


def calculate_all_metrics(y_true, y_pred, y_pred_proba=None, average='weighted'):
    """
    Calculate all required evaluation metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC)
        average: Averaging method for multi-class ('weighted', 'macro', 'micro')

    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}

    # Accuracy
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)

    # AUC Score
    try:
        if y_pred_proba is not None:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # Binary classification
                if len(y_pred_proba.shape) == 2:
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
            else:
                # Multi-class classification
                metrics['AUC'] = roc_auc_score(
                    y_true, y_pred_proba,
                    multi_class='ovr',
                    average=average
                )
        else:
            metrics['AUC'] = 0.0
    except Exception as e:
        metrics['AUC'] = 0.0

    # Precision
    metrics['Precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)

    # Recall
    metrics['Recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)

    # F1 Score
    metrics['F1'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Matthews Correlation Coefficient
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

    return metrics


def get_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        numpy array: Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)

