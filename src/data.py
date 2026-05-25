from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

import config


def load_20newsgroups():
    train_raw = fetch_20newsgroups(
        subset="train",
        categories=config.CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )
    test_raw = fetch_20newsgroups(
        subset="test",
        categories=config.CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_raw.data,
        train_raw.target.tolist(),
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=train_raw.target,
    )

    data = {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "val_texts": val_texts,
        "val_labels": val_labels,
        "test_texts": list(test_raw.data),
        "test_labels": test_raw.target.tolist(),
        "target_names": list(train_raw.target_names),
        "num_classes": len(train_raw.target_names),
    }

    print(
        f"Loaded 20 Newsgroups: "
        f"train={len(data['train_texts'])}, "
        f"val={len(data['val_texts'])}, "
        f"test={len(data['test_texts'])}, "
        f"classes={data['num_classes']}"
    )
    return data
if __name__ == "__main__":
    d = load_20newsgroups()
    print(f"\nSample document (truncated):\n{d['train_texts'][0][:200]}")
    print(f"\nLabel: {d['target_names'][d['train_labels'][0]]}")
