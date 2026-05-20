import os

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from tqdm.auto import tqdm
import wandb

import config


class TextOnlyDataset(Dataset):
    """Tokenize once up-front; collator handles masking dynamically."""

    def __init__(self, texts, tokenizer, max_length: int):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}


def pretrain_mlm(data: dict, save_path: str = None) -> str:
    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)
    save_path = save_path or os.path.join(config.MODELS_DIR, "task3_bert_mlm")

    # Use all texts (this is unsupervised, so test labels are irrelevant)
    texts = data["train_texts"] + data["val_texts"] + data["test_texts"]
    texts = [t for t in texts if len(t.strip()) > 50]
    print(f"  MLM pretraining on {len(texts)} documents")

    # Tokenizer + base model
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(config.BERT_MODEL).to(device)

    # Dataset + masking collator
    dataset = TextOnlyDataset(texts, tokenizer, config.BERT_MAX_LEN)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.MLM_PROBABILITY,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.MLM_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.MLM_LR)

    wandb.init(
        project=config.WANDB_PROJECT,
        name="task3_mlm_pretrain",
        mode=config.WANDB_MODE,
        config={
            "task": "3-mlm",
            "mlm_probability": config.MLM_PROBABILITY,
            "epochs": config.MLM_EPOCHS,
            "lr": config.MLM_LR,
            "batch_size": config.MLM_BATCH_SIZE,
            "max_len": config.BERT_MAX_LEN,
            "num_docs": len(texts),
        },
        reinit=True,
    )

    for epoch in range(1, config.MLM_EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in tqdm(loader, desc=f"MLM epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += out.loss.item()
            n_batches += 1
        avg = total_loss / max(1, n_batches)
        ppl = torch.exp(torch.tensor(avg)).item()
        print(f"  epoch {epoch}: MLM loss = {avg:.4f}   perplexity = {ppl:.2f}")
        wandb.log({"epoch": epoch, "mlm_loss": avg, "perplexity": ppl})

    # Save model + tokenizer in the same directory
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  domain-adapted BERT saved to: {save_path}")
    wandb.finish()
    return save_path
