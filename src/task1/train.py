"""
src/task1/train.py — Training and evaluation loop for the simple NN.

Reusable across Task 1b (Word2Vec input), 1c (TF-IDF input), and
1d (combined input). The NN architecture and hyperparameters stay
the same; only the input features change.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

import config
from src.task1.model import SimpleNN, VectorDataset
from src.utils.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
)


def _seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def train_simple_nn(X_train, y_train, X_val, y_val, X_test, y_test,
                    input_dim, num_classes, target_names,
                    run_name: str):
    """
    Full train-validate-test pipeline. Logs to W&B and saves the
    best-validation model checkpoint to outputs/models/.

    Returns a dict of test metrics: {accuracy, precision, recall, f1}.
    """
    _seed_all(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)

    # DataLoaders
    train_loader = DataLoader(VectorDataset(X_train, y_train),
                              batch_size=config.NN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(VectorDataset(X_val, y_val),
                            batch_size=config.NN_BATCH_SIZE)
    test_loader = DataLoader(VectorDataset(X_test, y_test),
                             batch_size=config.NN_BATCH_SIZE)

    # Model
    model = SimpleNN(input_dim=input_dim,
                     hidden_dim=config.NN_HIDDEN_DIM,
                     num_classes=num_classes,
                     dropout=config.NN_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.NN_LR,
                                 weight_decay=config.NN_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}")

    # W&B
    wandb.init(
        project=config.WANDB_PROJECT,
        name=run_name,
        mode=config.WANDB_MODE,
        config={
            "task": "task1",
            "run": run_name,
            "input_dim": input_dim,
            "hidden_dim": config.NN_HIDDEN_DIM,
            "lr": config.NN_LR,
            "batch_size": config.NN_BATCH_SIZE,
            "epochs": config.NN_EPOCHS,
            "dropout": config.NN_DROPOUT,
        },
        reinit=True,
    )

    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_path = os.path.join(config.MODELS_DIR, f"{run_name}_best.pth")

    for epoch in range(1, config.NN_EPOCHS + 1):
        tl, ta = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, va, _, _ = _evaluate(model, val_loader, criterion, device)
        train_losses.append(tl); val_losses.append(vl)
        train_accs.append(ta); val_accs.append(va)

        wandb.log({"epoch": epoch, "train_loss": tl, "train_acc": ta,
                   "val_loss": vl, "val_acc": va})

        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), best_path)

        if epoch == 1 or epoch % 5 == 0 or epoch == config.NN_EPOCHS:
            print(f"  epoch {epoch:3d}/{config.NN_EPOCHS} | "
                  f"train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f}")

    # Plots
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        title=run_name,
        save_path=os.path.join(config.PLOTS_DIR, f"{run_name}_curves.png"),
    )

    # Test using best validation checkpoint
    model.load_state_dict(torch.load(best_path, weights_only=True))
    _, _, preds, labels = _evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(labels, preds, target_names=target_names)
    plot_confusion_matrix(
        labels, preds, target_names=target_names,
        title=run_name,
        save_path=os.path.join(config.PLOTS_DIR, f"{run_name}_cm.png"),
    )

    wandb.log({f"test_{k}": v for k, v in metrics.items()})
    wandb.finish()
    return metrics
