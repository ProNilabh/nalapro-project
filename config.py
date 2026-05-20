"""
config.py — Central configuration for the NALAPRO project.

All hyperparameters and paths are defined here so individual scripts
do not need to be edited. Override per task by editing this file.
"""

import os
import torch

# ============================================================
# GENERAL
# ============================================================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output directories (created automatically)
OUTPUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Weights & Biases project name
WANDB_PROJECT = "nalapro-project"
WANDB_MODE = "online"  # "online", "offline", or "disabled"

# ============================================================
# DATASET (20 Newsgroups)
# ============================================================
CATEGORIES = None      # None = all 20 categories
VAL_SIZE = 0.2         # Fraction of train used for validation

# ============================================================
# TASK 1: WORD2VEC
# ============================================================
W2V_EMBED_DIM = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_WORKERS = 1
W2V_EPOCHS_TO_COMPARE = [1, 5, 20]   # For embedding visualization

# ============================================================
# TASK 1: TF-IDF
# ============================================================
TFIDF_MAX_FEATURES = 10000

# ============================================================
# TASK 1: SIMPLE NN
# ============================================================
NN_HIDDEN_DIM = 256
NN_LR = 1e-3
NN_BATCH_SIZE = 64
NN_EPOCHS = 30
NN_DROPOUT = 0.3
NN_WEIGHT_DECAY = 1e-4

# ============================================================
# TASK 2 & 3: BERT
# ============================================================
BERT_MODEL = "bert-base-uncased"
BERT_LR = 2e-5
BERT_BATCH_SIZE = 4
BERT_EPOCHS = 1
BERT_WARMUP_RATIO = 0.1
BERT_MAX_LEN = 64

# ============================================================
# TASK 3: MLM PRETRAINING
# ============================================================
MLM_PROBABILITY = 0.15
MLM_EPOCHS = 1
MLM_LR = 5e-5
MLM_BATCH_SIZE = 4

# ============================================================
# TASK 4: LLAMA-3
# ============================================================
# Default model (requires HF gated access). See docs/TASK4_GUIDE.md for alternatives.
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

LLAMA_MAX_NEW_TOKENS = 30
LLAMA_TEMPERATURE = 0.1
LLAMA_NUM_FEW_SHOT = 3        # Examples in few-shot prompt
LLAMA_EVAL_SAMPLES = 200      # How many test docs to evaluate (LLM is slow)
LLAMA_USE_4BIT = True         # 4-bit quantization to fit on small GPUs
