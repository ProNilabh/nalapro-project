"""
src/task3/finetune.py — Classification fine-tuning AFTER MLM pretraining.

Reuses src.task2.train.finetune_bert() but points the base_model
argument at the local MLM-pretrained checkpoint instead of the
official bert-base-uncased.
"""

from src.task2.train import finetune_bert


def finetune_after_mlm(data: dict, mlm_checkpoint: str):
    """Run the same fine-tuning pipeline as Task 2 but on the MLM model."""
    return finetune_bert(
        data,
        base_model=mlm_checkpoint,
        run_name="task3_bert_mlm_finetune",
        ignore_mismatched_sizes=True,   # MLM head ≠ classification head
    )
