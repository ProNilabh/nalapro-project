# 20 Newsgroups Text Classification — NALAPRO Project

Nilabh PANDEY for HSLU

> From classical word embeddings to fine-tuned large language models, a controlled comparison of four generations of NLP methods on the same benchmark.

This project classifies posts from the **20 Newsgroups** dataset into their topic categories. It benchmarks a progression of methods: classical features (Word2Vec / TF-IDF) → BERT fine-tuning → domain-adaptive BERT (MLM) → Llama-3 prompting → a parameter-efficient **QLoRA** fine-tuning bonus. All experiments are tracked with [Weights & Biases](https://wandb.ai/).

---

## 📌 What we're doing

The task is a classic, genuinely hard NLP benchmark: assign each forum post to exactly **one of 20 topic categories** spanning religion, computing, science, sports, politics, and for-sale ads. Several categories overlap heavily (three religion classes, three politics classes, a cluster of computing classes), and the random-guess baseline is just **5%**, so any real accuracy must be learned.

We hold the task fixed and vary the **method**, to answer: *how much does each generation of NLP technique actually buy us on the same problem?*

| Task | Method | Idea |
|------|--------|------|
| **Task 1** | Classical features + simple NN | Word2Vec, TF-IDF, and a combined representation fed to a feed-forward network |
| **Task 2** | BERT fine-tuning | Full supervised fine-tuning of a contextual transformer |
| **Task 3** | Domain-adaptive MLM → classify | Masked-language pre-training on in-domain text, *then* classification |
| **Task 4** | Llama-3 zero-shot & few-shot | An 8B instruction-tuned LLM with no task training |
| **Bonus** | Llama-3 QLoRA fine-tuning | Parameter-efficient fine-tuning (~0.2% of weights) |

---

## 📊 Dataset

- **20 Newsgroups** — 18,846 documents across 20 balanced classes
- **Splits:** 9,051 train · 2,263 validation · 7,532 test (stratified)
- **MLM corpus:** 17,886 in-domain documents (for Task 3 domain adaptation)
- **Preprocessing:** headers, footers, and quotes removed so models learn topical language rather than metadata artifacts; lowercasing, URL/email/number stripping, and stopword removal for the classical pipeline

---

## 🏆 Results

| Method | Accuracy | Macro-F1 | Training data |
|--------|:--------:|:--------:|:-------------:|
| Word2Vec + NN | 67.2% | 0.659 | 9,051 |
| TF-IDF + NN | 68.9% | 0.678 | 9,051 |
| Combined + NN | 70.2% | 0.687 | 9,051 |
| BERT fine-tune | 71.1% | 0.698 | 9,051 |
| **BERT + MLM** ⭐ | **71.5%** | **0.694** | 9,051 + 17,886 |
| Llama-3 zero-shot | 51.5% | 0.475 | 0 |
| Llama-3 few-shot (k=3) | 47.5% | 0.420 | 3 |
| Llama-3 QLoRA | 62.0% | 0.625 | ~2,000 |

*Random-guess baseline = 5%. The transformer results are the best configuration of each method (BERT/MLM: max_len 256, batch 8). LLMs evaluated on a fixed 200-document test sample for affordability and fair comparison. QLoRA accuracy is from the 100-docs/class × 1-epoch run; the repository notebook is configured for a stronger 150-docs/class × 2-epoch run (final training loss 0.25).*

### Key findings

- **Domain adaptation pays off.** MLM pre-training on in-domain text (Task 3) edged out plain BERT, with masked-LM perplexity dropping from **15.9 → 11.2**.
- **Combine your features.** Concatenating dense Word2Vec with sparse TF-IDF beat either representation alone.
- **Pretraining is powerful.** Llama-3 reached **51.5% with zero task-specific training**, rivaling simple trained models across 20 classes.
- **Prompt format > example count.** Few-shot *underperformed* zero-shot; the decisive fix was using Llama-3's native **chat template**, which cut unparseable outputs from 22.5% to ~1%.
- **Fine-tuning beats prompting.** QLoRA (tuning only ~0.2% of the weights) clearly beat zero/few-shot, though full BERT fine-tuning remained best at this data scale.
- **Compute matters.** The same BERT model gained ~7% accuracy (64.2% → 71.1%) purely by moving from a CPU laptop to a GPU.

Confusion matrices and exploratory plots are saved in [`outputs/plots/`](outputs/plots/).

---

## 📁 Repository structure

```
.
├── docs/                      # Project documentation & report
├── outputs/plots/             # Generated figures (EDA + confusion matrices)
├── src/                       # Task implementations
│   ├── task1/                 #   Word2Vec / TF-IDF + neural network
│   ├── task2/                 #   BERT fine-tuning
│   ├── task3/                 #   MLM domain adaptation → classification
│   └── task4/                 #   Llama-3 zero-shot / few-shot
├── Collab Train Notebook.ipynb   # Colab orchestration (runs the full pipeline + bonus)
├── config.py                  # Central hyper-parameters & paths
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🚀 Setup & usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
# For the Llama-3 tasks (GPU):
pip install bitsandbytes>=0.43.0 accelerate>=0.30.0 peft
```

### 2. Configure
Edit hyper-parameters (batch size, epochs, max sequence length, etc.) in **`config.py`**.

### 3. Run a task
Each task runs as a module:
```bash
python -m src.task1.run     # classical features + NN
python -m src.task2.run     # BERT fine-tuning
python -m src.task3.run     # MLM → classify
python -m src.task4.run     # Llama-3 zero-shot & few-shot
```

> **Tip:** Tasks 2–4 and the QLoRA bonus need a GPU. The full pipeline is most easily reproduced on Google Colab via **`Collab Train Notebook.ipynb`**, which sets the config, runs each task, and logs to Weights & Biases.

### 4. Authentication
The transformer tasks require:
- A **Hugging Face** token with access to the gated `meta-llama/Meta-Llama-3-8B-Instruct` model
- A **Weights & Biases** API key for experiment logging

Set these as environment variables (`HF_TOKEN`, `WANDB_API_KEY`) — **do not commit them to the repository**.

---

## 🛠️ Tech stack

`PyTorch` · `Hugging Face Transformers` · `PEFT` (LoRA/QLoRA) · `bitsandbytes` (4-bit quantization) · `scikit-learn` · `gensim` (Word2Vec) · `Weights & Biases` · `matplotlib` / `seaborn`

---

## 📈 Experiment tracking

All runs — hyperparameters, training curves, and final metrics — are logged to Weights & Biases for full reproducibility.

---
 
## 🤖 AI Usage Declaration
 
Generative AI tools were used as coding and writing aids during this project for debugging and refining text. All experimental design, model training, and reported results are the authors' own work; every metric in this README is produced by the project's own code and logged to Weights & Biases for verification.

---

## 👥 Authors

Nilabh Pandey — HSLU, NALAPRO course.
