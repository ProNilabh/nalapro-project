# 20 Newsgroups Text Classification (NALAPRO Project)

Nilabh Pandey - HSLU, NALAPRO course.

- **Code (GitHub):** https://github.com/ProNilabh/nalapro-project
- **Experiments (Weights & Biases):** https://wandb.ai/nilabhpandey7-hslu/nalapro-project
- **Report:** [`docs/report.pdf`](docs/report.pdf)

This project takes one text-classification task — sorting 20 Newsgroups posts into their topic — and solves it with four different families of methods, so I can see how much each step actually adds. The progression goes from classical word features (Word2Vec and TF-IDF) to a fine-tuned BERT, then a domain-adapted BERT, then a Llama-3 model used by prompting, and finally a cheaply fine-tuned Llama-3 using QLoRA as a bonus.

## The task

Each post has to be placed in exactly one of 20 topics, covering religion, computing, science, sports, politics, and for-sale ads. It is harder than it looks: several topics overlap a lot (three religion classes, three politics classes, and a cluster of computing classes), so they share much of the same vocabulary. With 20 roughly equal classes, random guessing only gets about 5% right, so any real accuracy has to be learned.

| Task | Method | Idea |
|------|--------|------|
| Task 1 | Classical features + small NN | Word2Vec, TF-IDF, and the two combined, fed to a two-layer network |
| Task 2 | BERT fine-tuning | Full supervised fine-tuning of a contextual transformer |
| Task 3 | MLM domain adaptation → classify | Masked-language pre-training on in-domain text, then classification |
| Task 4 | Llama-3 zero-shot & few-shot | An 8B instruction-tuned model with no task training |
| Bonus | Llama-3 QLoRA fine-tuning | Parameter-efficient fine-tuning (~0.2% of the weights) |

## Dataset

The data comes straight from scikit-learn's `fetch_20newsgroups`, so it is not stored in this repository.

- 18,846 documents over 20 roughly balanced classes (11,314 train + 7,532 test from scikit-learn).
- I split the training portion into 9,051 train and 2,263 validation, and keep the 7,532 test set untouched.
- Task 3 also uses 17,886 unlabelled documents for the masked-language-modeling stage.
- Headers, footers, and quoted replies are removed from every post so the models cannot cheat on metadata. The classical pipeline cleans further (lowercasing, stopword removal, stripping numbers, URLs, and emails); BERT and Llama-3 keep the text close to its natural form and rely on their own tokenizers.

## Results

| Method | Accuracy | Macro-F1 | Training data |
|--------|:--------:|:--------:|:-------------:|
| Word2Vec + NN | 67.2% | 0.659 | 9,051 |
| TF-IDF + NN | 68.9% | 0.678 | 9,051 |
| Combined + NN | 70.2% | 0.687 | 9,051 |
| BERT fine-tune | 71.1% | 0.698 | 9,051 |
| BERT + MLM | 71.5% | 0.694 | 9,051 + 17,886 |
| Llama-3 zero-shot | 51.5% | 0.475 | 0 |
| Llama-3 few-shot (k=3) | 47.5% | 0.420 | 3 |
| **Llama-3 QLoRA** ⭐ | **78.0%** | **0.746** | ~3,000 |

The transformer rows use the best configuration of each method (BERT and MLM at max_len 256). The three Llama-3 rows, including QLoRA, were scored on a fixed 200-document test sample to keep inference cheap and the comparison fair between them; the classical and BERT models were scored on the full 7,532-document test set. So the QLoRA number and the BERT numbers are not measured on exactly the same posts. The QLoRA run fine-tunes on about 3,000 documents (150 per class, 2 epochs).

### What I found

- Prompting alone is mediocre, but fine-tuning is not. Zero-shot Llama-3 only reached 51.5%, yet the same model fine-tuned with QLoRA — touching about 0.2% of its weights — jumped to 78.0% and became the best model in the project, ahead of the BERT variants.
- Domain adaptation helps a little. Continuing BERT's masked-language pre-training on the newsgroup text first (Task 3) edged past plain BERT, and its perplexity dropped from 15.9 to 11.2.
- Combining features beats either one. Concatenating dense Word2Vec with sparse TF-IDF did better than either on its own.
- Prompt format matters more than examples. Few-shot actually did worse than zero-shot. The change that really mattered was switching to Llama-3's chat template, which cut unparseable answers from about 22.5% to roughly 1%.
- Compute matters. The same BERT gained about seven points (64.2% → 71.1%) just by moving from a CPU laptop to a GPU with a longer sequence length and more epochs.

The confusion matrices and the exploratory plots are in [`outputs/plots/`](outputs/plots/).

## Repository structure

```
.
├── docs/                         # Documentation and the final report
├── outputs/plots/                # Figures (EDA + confusion matrices)
├── src/                          # Task implementations
│   ├── task1/                    #   Word2Vec / TF-IDF + neural network
│   ├── task2/                    #   BERT fine-tuning
│   ├── task3/                    #   MLM domain adaptation → classification
│   └── task4/                    #   Llama-3 zero-shot / few-shot
├── Collab Train Notebook.ipynb   # Colab notebook (full pipeline + bonus)
├── config.py                     # Hyper-parameters and paths
├── requirements.txt              # Python dependencies
└── README.md
```

## Setup and usage

Install dependencies:

```bash
pip install -r requirements.txt
# extra packages for the Llama-3 tasks (need a GPU):
pip install "bitsandbytes>=0.43.0" "accelerate>=0.30.0" peft
```

Hyper-parameters (batch size, epochs, max sequence length, and so on) live in `config.py`. Each task runs as a module:

```bash
python -m src.task1.run     # classical features + NN
python -m src.task2.run     # BERT fine-tuning
python -m src.task3.run     # MLM → classify
python -m src.task4.run     # Llama-3 zero-shot & few-shot
```

Tasks 2–4 and the QLoRA bonus need a GPU. The easiest way to reproduce everything is the `Collab Train Notebook.ipynb` on Google Colab, which sets the config, runs each task, and logs to Weights & Biases.

The transformer tasks need a Hugging Face token with access to the gated `meta-llama/Meta-Llama-3-8B-Instruct` model and a Weights & Biases API key. Set them as environment variables (`HF_TOKEN`, `WANDB_API_KEY`) and do not commit them to the repo.

## Tech stack

PyTorch, Hugging Face Transformers, PEFT (LoRA/QLoRA), bitsandbytes (4-bit quantization), scikit-learn, gensim (Word2Vec), Weights & Biases, and matplotlib/seaborn.

## AI usage declaration

I used generative AI tools as an assistant for debugging code, explaining unfamiliar library parts, and improving the wording of the documentation and report. I did not use AI to produce my results. The experimental design, the training, and every number reported here are my own work, produced by my own code and logged to Weights & Biases. I understand the whole project and can explain any part of it.
