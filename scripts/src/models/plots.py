"""
Plotting utilities for model evaluation.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score


def plot_roc_prc(y_true: np.ndarray, y_prob: np.ndarray, ci: float = 0.95):
    """Plot ROC and PRC"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plot ROC curve
    axs[0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axs[0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title(f'Receiver Operating Characteristic (AUC = {roc_auc:.4f})')
    axs[0].legend(loc="lower right")

    # Plot PRC curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    axs[1].plot(recall, precision, color='blue', lw=2, label=f'PRC (AP = {average_precision:.2f})')
    axs[1].fill_between(recall, precision, alpha=0.2, color='blue')
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_title(f'Precision-Recall Curve (AUPRC = {average_precision:.4f})')
    axs[1].legend(loc="lower left")

    plt.show()

