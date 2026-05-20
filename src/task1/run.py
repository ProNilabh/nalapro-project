import numpy as np

import config
from src.data import load_20newsgroups
from src.utils.preprocessing import preprocess_to_tokens, preprocess_to_strings
from src.task1.embeddings import (
    train_word2vec, docs_to_matrix,
    fit_tfidf, texts_to_tfidf,
)
from src.task1.train import train_simple_nn
from src.task1.visualize import run_embedding_visualization


def main():
    # 1a. Load + preprocess
    print("\n" + "=" * 60)
    print("TASK 1a — Loading and preprocessing")
    print("=" * 60)
    data = load_20newsgroups()

    print("\n[1a] tokenizing train/val/test for Word2Vec...")
    train_tokens = preprocess_to_tokens(data["train_texts"])
    val_tokens = preprocess_to_tokens(data["val_texts"])
    test_tokens = preprocess_to_tokens(data["test_texts"])
    all_tokens = train_tokens + val_tokens + test_tokens

    print("[1a] preparing strings for TF-IDF...")
    train_strs = preprocess_to_strings(data["train_texts"])
    val_strs = preprocess_to_strings(data["val_texts"])
    test_strs = preprocess_to_strings(data["test_texts"])

    results = {}

    # 1b. Word2Vec → simple NN
    print("\n" + "=" * 60)
    print("TASK 1b — Word2Vec + simple NN")
    print("=" * 60)
    w2v = train_word2vec(all_tokens, epochs=config.W2V_EPOCHS_TO_COMPARE[-1])

    Xtr = docs_to_matrix(train_tokens, w2v)
    Xva = docs_to_matrix(val_tokens, w2v)
    Xte = docs_to_matrix(test_tokens, w2v)
    print(f"  feature shapes: train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

    results["w2v_nn"] = train_simple_nn(
        Xtr, data["train_labels"],
        Xva, data["val_labels"],
        Xte, data["test_labels"],
        input_dim=config.W2V_EMBED_DIM,
        num_classes=data["num_classes"],
        target_names=data["target_names"],
        run_name="task1b_w2v_nn",
    )

    # 1b extra: embedding visualizations
    print("\n" + "=" * 60)
    print("TASK 1b extra — Embedding visualizations (1, 5, 20, 50 epochs)")
    print("=" * 60)
    run_embedding_visualization(all_tokens)

    # 1c. TF-IDF → simple NN
    print("\n" + "=" * 60)
    print("TASK 1c — TF-IDF + simple NN")
    print("=" * 60)
    tfidf = fit_tfidf(train_strs)
    Xtr_t = texts_to_tfidf(train_strs, tfidf)
    Xva_t = texts_to_tfidf(val_strs, tfidf)
    Xte_t = texts_to_tfidf(test_strs, tfidf)
    print(f"  feature shapes: train={Xtr_t.shape} val={Xva_t.shape} test={Xte_t.shape}")

    results["tfidf_nn"] = train_simple_nn(
        Xtr_t, data["train_labels"],
        Xva_t, data["val_labels"],
        Xte_t, data["test_labels"],
        input_dim=Xtr_t.shape[1],
        num_classes=data["num_classes"],
        target_names=data["target_names"],
        run_name="task1c_tfidf_nn",
    )

    # 1d. Extra experiment — concatenated W2V + TF-IDF
    print("\n" + "=" * 60)
    print("TASK 1d — Combined W2V + TF-IDF + simple NN")
    print("=" * 60)
    Xtr_c = np.concatenate([Xtr, Xtr_t], axis=1)
    Xva_c = np.concatenate([Xva, Xva_t], axis=1)
    Xte_c = np.concatenate([Xte, Xte_t], axis=1)
    print(f"  combined dim: {Xtr_c.shape[1]} "
          f"(W2V={Xtr.shape[1]} + TF-IDF={Xtr_t.shape[1]})")

    results["combined_nn"] = train_simple_nn(
        Xtr_c, data["train_labels"],
        Xva_c, data["val_labels"],
        Xte_c, data["test_labels"],
        input_dim=Xtr_c.shape[1],
        num_classes=data["num_classes"],
        target_names=data["target_names"],
        run_name="task1d_combined_nn",
    )

    # Summary
    print("\n" + "=" * 60)
    print("TASK 1 — Summary")
    print("=" * 60)
    print(f"{'Run':<20}{'Acc':>10}{'F1':>10}")
    for name, m in results.items():
        print(f"{name:<20}{m['accuracy']:>10.4f}{m['f1']:>10.4f}")

    return results


if __name__ == "__main__":
    main()
