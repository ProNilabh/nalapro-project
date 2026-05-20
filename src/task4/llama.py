import random

import numpy as np
import torch
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

import config


# Model loading (supports optional 4-bit quantization)

def load_llama():
    print(f"  loading {config.LLAMA_MODEL}...")

    quant_config = None
    if config.LLAMA_USE_4BIT and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            print("  using 4-bit quantization (bitsandbytes)")
        except ImportError:
            print("  bitsandbytes not installed — falling back to fp16")

    tokenizer = AutoTokenizer.from_pretrained(config.LLAMA_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": "auto"}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(config.LLAMA_MODEL, **model_kwargs)
    model.eval()
    return model, tokenizer


# Prompt construction

def _category_list(target_names) -> str:
    return "\n".join(f"  - {name}" for name in target_names)


def zero_shot_prompt(text: str, target_names) -> str:
    """A direct classification prompt with no examples."""
    return (
        "You are a text classifier. Read the newsgroup post and answer with "
        "EXACTLY ONE of the category names listed below — nothing else.\n\n"
        f"Categories:\n{_category_list(target_names)}\n\n"
        f"Post:\n\"\"\"{text[:1500]}\"\"\"\n\n"
        "Answer with only the category name.\n"
        "Category:"
    )


def few_shot_prompt(text: str, target_names, examples) -> str:
    """Same as zero-shot, but with labeled in-context examples first."""
    ex_str = ""
    for ex_text, ex_label in examples:
        ex_str += (f"\nPost: \"\"\"{ex_text[:300]}\"\"\"\n"
                   f"Category: {ex_label}\n")
    return (
        "You are a text classifier. Read the newsgroup post and answer with "
        "EXACTLY ONE of the category names listed below — nothing else.\n\n"
        f"Categories:\n{_category_list(target_names)}\n\n"
        f"Examples:\n{ex_str}\n"
        f"Now classify this post:\n"
        f"Post: \"\"\"{text[:1000]}\"\"\"\n\n"
        "Answer with only the category name.\n"
        "Category:"
    )


# Response parsing

def parse_response(response: str, target_names) -> int:
    """
    Map the model's free-text answer back to a category index.
    Uses a two-pass fuzzy match because LLMs rarely produce the exact
    label verbatim.
    """
    r = response.strip().lower()

    # Pass 1: full category name appears in the response
    for i, name in enumerate(target_names):
        if name.lower() in r:
            return i

    # Pass 2: a distinctive sub-token of the category name appears
    for i, name in enumerate(target_names):
        parts = name.lower().replace(".", " ").replace("-", " ").split()
        if any(len(p) > 3 and p in r for p in parts):
            return i

    return -1   # unparseable


# Few-shot example selection
def pick_few_shot_examples(train_texts, train_labels, target_names, k: int):
    """
    Pick `k` (text, label_name) examples for in-context learning.
    Strategy: one example per class, then shuffle and keep first k.
    """
    seen = {}
    for idx, label in enumerate(train_labels):
        if label not in seen:
            seen[label] = idx
    examples = [(train_texts[i], target_names[lbl]) for lbl, i in seen.items()]
    random.shuffle(examples)
    return examples[:k]


# Experiment runner
def run_llama_experiment(model, tokenizer, data: dict, mode: str):
    """
    Run zero-shot or few-shot classification with the loaded model.

    Parameters
    ----------
    mode : "zero_shot" or "few_shot"

    Returns
    -------
    (preds, labels) — predicted and ground-truth label indices.
    """
    assert mode in ("zero_shot", "few_shot")
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    test_texts = data["test_texts"]
    test_labels = data["test_labels"]
    target_names = data["target_names"]

    # Sample a manageable subset (LLM inference is slow)
    n = min(config.LLAMA_EVAL_SAMPLES, len(test_texts))
    indices = random.sample(range(len(test_texts)), n)

    examples = []
    if mode == "few_shot":
        examples = pick_few_shot_examples(
            data["train_texts"], data["train_labels"],
            target_names, config.LLAMA_NUM_FEW_SHOT,
        )
        print(f"  using {len(examples)} in-context examples")

    preds, labels, unparseable = [], [], 0

    for idx in tqdm(indices, desc=f"llama-{mode}"):
        text = test_texts[idx]
        true_label = test_labels[idx]

        prompt = (zero_shot_prompt(text, target_names)
                  if mode == "zero_shot"
                  else few_shot_prompt(text, target_names, examples))

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3500
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=config.LLAMA_MAX_NEW_TOKENS,
                temperature=config.LLAMA_TEMPERATURE,
                do_sample=False,           # deterministic
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode ONLY the newly generated tokens
        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        pred = parse_response(response, target_names)
        if pred == -1:
            unparseable += 1
            pred = 0   # fallback so metrics can still be computed
        preds.append(pred)
        labels.append(true_label)

    print(f"  unparseable: {unparseable}/{n} ({100*unparseable/n:.1f}%)")
    return preds, labels
