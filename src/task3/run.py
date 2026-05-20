import config
from src.data import load_20newsgroups
from src.task3.mlm import pretrain_mlm
from src.task3.finetune import finetune_after_mlm


def main():
    data = load_20newsgroups()

    print("\n" + "=" * 60)
    print("TASK 3 Stage 1 — MLM pretraining")
    print("=" * 60)
    mlm_dir = pretrain_mlm(data)

    print("\n" + "=" * 60)
    print("TASK 3 Stage 2 — Fine-tune classification on MLM-adapted BERT")
    print("=" * 60)
    metrics = finetune_after_mlm(data, mlm_checkpoint=mlm_dir)

    print(f"\nFinal Task 3 test accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Task 3 test F1:       {metrics['f1']:.4f}")
    print("\nCompare these to Task 2 to see the effect of MLM domain adaptation.")
    return metrics


if __name__ == "__main__":
    main()
