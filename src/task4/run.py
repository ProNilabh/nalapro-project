"""
src/task4/run.py — Entry point for Task 4 (Llama-3 zero-shot / few-shot).

Important: this task does NOT fine-tune the LLM. We construct prompts
and read the LLM's text response, mapping it back to one of the 20
category names.

Before running, read docs/TASK4_GUIDE.md for:
  - how to get Hugging Face access to Meta-Llama-3-8B-Instruct
  - GPU / memory requirements
  - smaller-model alternatives if your machine cannot run Llama-3-8B

Usage:
    python -m src.task4.run
"""

import os

import wandb

import config
from src.data import load_20newsgroups
from src.task4.llama import load_llama, run_llama_experiment
from src.utils.evaluation import compute_metrics, plot_confusion_matrix


def main():
    print("\n" + "=" * 60)
    print(f"TASK 4 — Llama-3 zero-shot / few-shot ({config.LLAMA_MODEL})")
    print("=" * 60)

    data = load_20newsgroups()
    model, tokenizer = load_llama()

    wandb.init(
        project=config.WANDB_PROJECT,
        name="task4_llama",
        mode=config.WANDB_MODE,
        config={
            "task": "4-llama",
            "model": config.LLAMA_MODEL,
            "eval_samples": config.LLAMA_EVAL_SAMPLES,
            "few_shot_k": config.LLAMA_NUM_FEW_SHOT,
            "temperature": config.LLAMA_TEMPERATURE,
            "use_4bit": config.LLAMA_USE_4BIT,
        },
        reinit=True,
    )

    # -------- Zero-shot --------
    print("\n[zero-shot]")
    preds, labels = run_llama_experiment(model, tokenizer, data, mode="zero_shot")
    zs = compute_metrics(labels, preds, target_names=data["target_names"])
    plot_confusion_matrix(
        labels, preds, target_names=data["target_names"],
        title="Llama-3 Zero-Shot",
        save_path=os.path.join(config.PLOTS_DIR, "task4_zeroshot_cm.png"),
    )
    wandb.log({"zs_accuracy": zs["accuracy"], "zs_f1": zs["f1"]})

    # -------- Few-shot --------
    print(f"\n[few-shot, k={config.LLAMA_NUM_FEW_SHOT}]")
    preds, labels = run_llama_experiment(model, tokenizer, data, mode="few_shot")
    fs = compute_metrics(labels, preds, target_names=data["target_names"])
    plot_confusion_matrix(
        labels, preds, target_names=data["target_names"],
        title="Llama-3 Few-Shot",
        save_path=os.path.join(config.PLOTS_DIR, "task4_fewshot_cm.png"),
    )
    wandb.log({"fs_accuracy": fs["accuracy"], "fs_f1": fs["f1"]})

    wandb.finish()

    print("\n" + "=" * 60)
    print("TASK 4 — Summary")
    print("=" * 60)
    print(f"Zero-shot — accuracy {zs['accuracy']:.4f}  F1 {zs['f1']:.4f}")
    print(f"Few-shot  — accuracy {fs['accuracy']:.4f}  F1 {fs['f1']:.4f}")

    return {"zero_shot": zs, "few_shot": fs}


if __name__ == "__main__":
    main()
