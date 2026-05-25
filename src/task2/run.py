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
