"""
src/task2/train.py — BERT fine-tuning loop.

Reusable building blocks:
    bert_train_one_epoch()
    bert_evaluate()
    finetune_bert()

`finetune_bert()` is called by Task 2 directly (from a pretrained
bert-base-uncased) and by Task 3 (from a domain-adapted MLM model).
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm
import wandb

import config
from src.task2.bert_data import BertClassificationDataset, get_tokenizer
from src.utils.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
)


# -----------------------------------------------------------------
# epoch helpers
# -----------------------------------------------------------------

def bert_train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc="train", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += out.loss.item() * labels.size(0)
        correct += (out.logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def bert_evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            total_loss += out.loss.item() * labels.size(0)
            preds = out.logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# -----------------------------------------------------------------
# Full fine-tuning pipeline
# -----------------------------------------------------------------

def finetune_bert(data: dict, base_model: str, run_name: str,
                  ignore_mismatched_sizes: bool = False):
    """
    Fine-tune a BERT-family model with a classification head.

    Parameters
    ----------
    data : dict from src.data.load_20newsgroups()
    base_model : HF model name OR local path
        For Task 2: "bert-base-uncased"
        For Task 3: path to MLM-pretrained checkpoint
    run_name : descriptive name for W&B + saved files
    ignore_mismatched_sizes : True when loading from an MLM checkpoint
        (the MLM head is different from the classification head).

    Returns
    -------
    dict of test metrics.
    """
    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)

    # Tokenizer + model
    tokenizer = get_tokenizer(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=data["num_classes"],
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    ).to(device)

    # Datasets
    train_ds = BertClassificationDataset(
        data["train_texts"], data["train_labels"], tokenizer, config.BERT_MAX_LEN
    )
    val_ds = BertClassificationDataset(
        data["val_texts"], data["val_labels"], tokenizer, config.BERT_MAX_LEN
    )
    test_ds = BertClassificationDataset(
        data["test_texts"], data["test_labels"], tokenizer, config.BERT_MAX_LEN
    )

    train_loader = DataLoader(train_ds, batch_size=config.BERT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BERT_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=config.BERT_BATCH_SIZE)

    # Optimizer + linear-warmup scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.BERT_LR, weight_decay=0.01
    )
    total_steps = len(train_loader) * config.BERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.BERT_WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    # W&B
    wandb.init(
        project=config.WANDB_PROJECT,
        name=run_name,
        mode=config.WANDB_MODE,
        config={
            "task": run_name,
            "base_model": base_model,
            "lr": config.BERT_LR,
            "epochs": config.BERT_EPOCHS,
            "batch_size": config.BERT_BATCH_SIZE,
            "max_len": config.BERT_MAX_LEN,
            "warmup_ratio": config.BERT_WARMUP_RATIO,
        },
        reinit=True,
    )

    # Training loop with checkpointing
    tls, vls, tas, vas = [], [], [], []
    best_val_acc = 0.0
    best_dir = os.path.join(config.MODELS_DIR, f"{run_name}_best")

    for epoch in range(1, config.BERT_EPOCHS + 1):
        print(f"\n[epoch {epoch}/{config.BERT_EPOCHS}]")
        tl, ta = bert_train_one_epoch(model, train_loader, optimizer, scheduler, device)
        vl, va, _, _ = bert_evaluate(model, val_loader, device)
        tls.append(tl); vls.append(vl); tas.append(ta); vas.append(va)
        wandb.log({"epoch": epoch, "train_loss": tl, "train_acc": ta,
                   "val_loss": vl, "val_acc": va})
        print(f"  train {tl:.4f}/{ta:.4f}   val {vl:.4f}/{va:.4f}")
        if va > best_val_acc:
            best_val_acc = va
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    # Plots
    plot_training_curves(
        tls, vls, tas, vas,
        title=run_name,
        save_path=os.path.join(config.PLOTS_DIR, f"{run_name}_curves.png"),
    )

    # Final test using best checkpoint
    print("\n[test] loading best checkpoint and evaluating...")
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    _, _, preds, labels = bert_evaluate(model, test_loader, device)
    metrics = compute_metrics(labels, preds, target_names=data["target_names"])
    plot_confusion_matrix(
        labels, preds, target_names=data["target_names"],
        title=run_name,
        save_path=os.path.join(config.PLOTS_DIR, f"{run_name}_cm.png"),
    )

    wandb.log({f"test_{k}": v for k, v in metrics.items()})
    wandb.finish()
    return metrics
