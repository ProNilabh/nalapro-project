"""
src/utils/evaluation.py — Shared evaluation and plotting utilities.

Provides:
  - compute_metrics(): accuracy, precision, recall, F1
  - plot_confusion_matrix(): raw + normalized CM heatmaps
  - plot_training_curves(): loss + accuracy across epochs
  - plot_comparison_bar(): final cross-model bar chart
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config


def compute_metrics(y_true, y_pred, target_names=None) -> dict:
    """Compute accuracy + macro precision/recall/F1 and print a report."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred,
                                target_names=target_names, zero_division=0))
    print(f"Accuracy: {acc:.4f} | Precision: {p:.4f} | "
          f"Recall: {r:.4f} | F1: {f1:.4f}")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, target_names, title, save_path):
    """Side-by-side raw + normalized confusion matrices."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=axes[0])
    axes[0].set_title(f"{title} — Counts")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=axes[1])
    axes[1].set_title(f"{title} — Normalized")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         title, save_path):
    """Two subplots: loss curve and accuracy curve over epochs."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, "b-o", label="Train", markersize=3)
    ax1.plot(epochs, val_losses, "r-o", label="Val", markersize=3)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-o", label="Train", markersize=3)
    ax2.plot(epochs, val_accs, "r-o", label="Val", markersize=3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} — Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def plot_comparison_bar(results, metric="accuracy", title="Model Comparison",
                        save_path=None):
    """Bar chart across model results dicts."""
    models = list(results.keys())
    values = [results[m][metric] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, values, color=sns.color_palette("viridis", len(models)))
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, f"comparison_{metric}.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")
