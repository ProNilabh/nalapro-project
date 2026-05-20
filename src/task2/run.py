"""
src/task2/run.py — Entry point for Task 2: fine-tune BERT-base.

Usage:
    python -m src.task2.run

Parameters you can experiment with (set in config.py):
    BERT_LR              — 2e-5 is the standard; try 3e-5 or 5e-5
    BERT_BATCH_SIZE      — 16 fits 8GB GPU; 32 needs ~14GB
    BERT_EPOCHS          — 3-5; more epochs risk overfitting
    BERT_MAX_LEN         — 128 (fast) vs 256 (better) vs 512 (slow)
    BERT_WARMUP_RATIO    — 0.05 to 0.1
"""

import config
from src.data import load_20newsgroups
from src.task2.train import finetune_bert


def main():
    print("\n" + "=" * 60)
    print("TASK 2 — Fine-tune bert-base-uncased")
    print("=" * 60)
    data = load_20newsgroups()
    metrics = finetune_bert(
        data,
        base_model=config.BERT_MODEL,
        run_name="task2_bert_finetune",
    )
    print(f"\nFinal Task 2 test accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Task 2 test F1:       {metrics['f1']:.4f}")
    return metrics


if __name__ == "__main__":
    main()
