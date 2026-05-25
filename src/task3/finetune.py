from src.task2.train import finetune_bert

def finetune_after_mlm(data: dict, mlm_checkpoint: str):
    return finetune_bert(
        data,
        base_model=mlm_checkpoint,
        run_name="task3_bert_mlm_finetune",
        ignore_mismatched_sizes=True,   # MLM head ≠ classification head
    )
