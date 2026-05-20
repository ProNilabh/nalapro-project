# How to Run

Each task is an independent module that you launch with `python -m`. They produce plots in `outputs/plots/`, model checkpoints in `outputs/models/`, and live experiment logs at https://wandb.ai/<your-username>/nalapro-project.

> **Before running anything**, complete the setup in `SETUP.md`.

## Task 1 — Simple Neural Network

```bash
python -m src.task1.run
```

This runs **all four sub-tasks in order**:
- **1a**: preprocess train / val / test data
- **1b**: train Word2Vec (1, 5, 20, 50 epochs) + simple NN + visualizations
- **1c**: TF-IDF + simple NN (same architecture as 1b)
- **1d**: combined W2V + TF-IDF + simple NN

**Wall time**: ~10–15 min on CPU. The Word2Vec retraining at 4 epoch counts dominates.

**Outputs:**
```
outputs/plots/
├── task1b_w2v_nn_curves.png         loss/accuracy of W2V+NN
├── task1b_w2v_nn_cm.png             confusion matrix
├── task1b_tsne_by_epoch.png         ⭐ 2D embedding visualization at 1/5/20/50 epochs
├── task1b_cosine_heatmaps.png       ⭐ word similarity at each epoch count
├── task1b_vector_drift.png          ⭐ how much vectors moved over training
├── task1c_tfidf_nn_curves.png       TF-IDF training curves
├── task1c_tfidf_nn_cm.png
├── task1d_combined_nn_curves.png    combined-features training curves
└── task1d_combined_nn_cm.png
```

The three starred plots directly answer Task 1b's "Did the vectors change in the space? What was the effect of training?"

## Task 2 — BERT Fine-Tuning

```bash
python -m src.task2.run
```

**Wall time**: ~25 min on a recent GPU (RTX 30/40-series), 2–4 hours on CPU.

**Reduce time if needed** — edit `config.py`:
```python
BERT_EPOCHS = 2        # was 4
BERT_MAX_LEN = 128     # was 256
BERT_BATCH_SIZE = 32   # if you have GPU memory
```

**Outputs:**
```
outputs/plots/task2_bert_finetune_curves.png
outputs/plots/task2_bert_finetune_cm.png
outputs/models/task2_bert_finetune_best/        full HF checkpoint (config + weights)
```

**What to experiment with for your report (Task 2 asks for this explicitly):**
- learning rate (1e-5, 2e-5, 3e-5, 5e-5)
- max sequence length (128 vs 256 vs 512)
- batch size (8 vs 16 vs 32)
- number of epochs (2 vs 4 vs 6)
- warmup ratio (0.0 vs 0.1)

Run several configurations by editing `config.py` between runs. W&B will keep them separate so you can compare.

## Task 3 — BERT MLM → Fine-Tune

```bash
python -m src.task3.run
```

This runs **two stages back-to-back**:
1. MLM pretraining — unsupervised, ~30 min on GPU
2. Classification fine-tuning — same as Task 2, ~25 min on GPU

**Wall time**: ~1 hour on GPU.

**Outputs:**
```
outputs/models/task3_bert_mlm/                  domain-adapted BERT (after MLM)
outputs/models/task3_bert_mlm_finetune_best/    final classifier
outputs/plots/task3_bert_mlm_finetune_curves.png
outputs/plots/task3_bert_mlm_finetune_cm.png
```

You can also run the two stages separately:
```python
# Just MLM
python -c "from src.data import load_20newsgroups; from src.task3.mlm import pretrain_mlm; pretrain_mlm(load_20newsgroups())"

# Just fine-tuning (after MLM is done)
python -c "from src.data import load_20newsgroups; from src.task3.finetune import finetune_after_mlm; finetune_after_mlm(load_20newsgroups(), 'outputs/models/task3_bert_mlm')"
```

## Task 4 — Llama-3 Zero-Shot & Few-Shot

```bash
python -m src.task4.run
```

**Read `TASK4_GUIDE.md` first** — there are multiple ways to set this up depending on your hardware.

**Wall time**: 20-60 min for 200 test samples, depending on hardware and model size.

**Outputs:**
```
outputs/plots/task4_zeroshot_cm.png
outputs/plots/task4_fewshot_cm.png
```

## Running Everything

```bash
python -m src.task1.run
python -m src.task2.run
python -m src.task3.run
python -m src.task4.run
```

Each task is independent — you don't have to run them in order. If a task fails, the others still work.

## Customizing Hyperparameters

All knobs live in `config.py`. Edit values there and rerun the task script. Common changes:

```python
# Smaller, faster Task 1
NN_EPOCHS = 10               # was 30
W2V_EPOCHS_TO_COMPARE = [1, 5, 20]   # was [1, 5, 20, 50]

# Smaller, faster Task 2/3
BERT_EPOCHS = 2
BERT_MAX_LEN = 128

# Smaller Task 4 evaluation
LLAMA_EVAL_SAMPLES = 50      # was 200
LLAMA_NUM_FEW_SHOT = 1       # was 3

# Disable W&B entirely
WANDB_MODE = "disabled"
```

## Monitoring Runs

While a script is running, visit https://wandb.ai/<your-username>/nalapro-project to see live training curves. Each `run_name` becomes a separate W&B run, so you can overlay multiple experiments to compare.

## Re-running with the same artifacts

By default each run overwrites `outputs/models/<run_name>_best/`. If you want to keep multiple runs, rename `outputs/` between runs, or edit the `run_name=` argument in the entry-point scripts.
