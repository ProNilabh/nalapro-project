# Setup Guide

This guide walks you through preparing your machine to run the project.

## 1. Prerequisites

You need:
- **Python 3.10 or higher** (`python --version` to check)
- **Git** (`git --version`)
- A free **Weights & Biases** account: https://wandb.ai/signup
- (Task 4 only) A free **Hugging Face** account: https://huggingface.co/join

### Hardware

| Task | Minimum | Recommended |
|------|---------|-------------|
| Task 1 (NN + W2V/TF-IDF) | Any CPU, 4 GB RAM | Any laptop |
| Task 2 (BERT fine-tune)  | 8 GB RAM, 4 GB GPU VRAM | 16 GB GPU |
| Task 3 (MLM + fine-tune) | Same as Task 2 | Same as Task 2 |
| Task 4 (Llama-3-8B)      | 16 GB GPU VRAM (4-bit) | 24 GB GPU |

**No GPU?** Task 1 still runs fine on CPU. Tasks 2-3 work on CPU but take hours. For Task 4 see `TASK4_GUIDE.md` for alternatives.

## 2. Clone the Repository

```bash
git clone https://github.com/<your-username>/nalapro-project.git
cd nalapro-project
```

## 3. Create a Virtual Environment

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your terminal prompt.

## 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs PyTorch, Transformers, Gensim, scikit-learn, NLTK, W&B, etc.

### Optional: PyTorch with GPU support

The default `pip install torch` gives you a CPU-only build on some systems. If you have an NVIDIA GPU and CUDA installed, install the matching GPU build instead:

```bash
# CUDA 12.1 example — check https://pytorch.org for your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is detected:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Optional: bitsandbytes (Task 4 4-bit quantization)

Only needed for Task 4 with a GPU smaller than ~24 GB. Uncomment the lines in `requirements.txt`:

```bash
pip install bitsandbytes peft
```

bitsandbytes only works on Linux/Windows with NVIDIA CUDA. On macOS, skip this — Task 4 will run in fp16 instead (and may not fit on M1/M2 unless you use a smaller model — see `TASK4_GUIDE.md`).

## 5. Log in to Weights & Biases

```bash
wandb login
```

It will open a browser. Paste your W&B API key (found at https://wandb.ai/authorize) when prompted.

**Don't want W&B?** Edit `config.py` and set:
```python
WANDB_MODE = "disabled"
```
The code will run normally, but no experiment tracking.

## 6. (Task 4 only) Hugging Face Login

See **`TASK4_GUIDE.md`** for the full Llama-3 setup. Short version:

1. Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Click "Access repository" and accept Meta's license (free, takes a minute)
3. Create an access token at https://huggingface.co/settings/tokens
4. Log in from the terminal:
   ```bash
   huggingface-cli login
   ```

## 7. First Run — Sanity Check

Test data loading:
```bash
python -m src.data
```
You should see:
```
Loaded 20 Newsgroups: train=9051, val=2263, test=7532, classes=20
```
The first time scikit-learn downloads the dataset to `~/scikit_learn_data/` (about 14 MB).

Test preprocessing:
```bash
python -m src.utils.preprocessing
```

If both work, you're ready to run the full tasks. See **`HOW_TO_RUN.md`** next.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: nltk` | Make sure your virtualenv is active before running |
| `OSError: Can't load tokenizer for 'bert-base-uncased'` | Internet required for first run; HF downloads models to `~/.cache/huggingface/` |
| `CUDA out of memory` (Task 2/3) | Reduce `BERT_BATCH_SIZE` to 8 or `BERT_MAX_LEN` to 128 in `config.py` |
| `bitsandbytes` import error | Reinstall with the right CUDA version, OR set `LLAMA_USE_4BIT = False` |
| W&B prompts for an API key in a script | Either run `wandb login` once, or set `WANDB_MODE = "disabled"` |
| `Trying to access gated repo` | You haven't accepted Llama's license yet — see Task 4 guide |
