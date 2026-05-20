# Task 4 — Llama-3 Zero-Shot & Few-Shot Guide

Task 4 uses a large language model (LLM) **without fine-tuning** — only prompting. This guide walks you through every option for running it, from the easiest to the most demanding.

## What Task 4 actually does

1. Load a pretrained instruct-tuned LLM (Meta-Llama-3-8B-Instruct by default).
2. For each test document, build a **prompt**:
   - **Zero-shot**: "Here is a post. Pick one of these 20 categories. Post: ..."
   - **Few-shot**: same, but with 3 labeled examples first.
3. Generate the model's text answer (e.g. "comp.graphics").
4. Parse the answer back into a category index.
5. Compute accuracy / F1 / confusion matrix.

**No gradient updates. No training.** This is purely inference.

## Option A — Run Llama-3-8B locally with a GPU (recommended)

### Step 1: Get access to Llama-3

Llama-3 is a "gated" model. You must accept Meta's license to download it (free, no review):

1. Create a free Hugging Face account at https://huggingface.co/join
2. Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
3. Click **"Expand to review and access"** at the top
4. Fill in the short form and accept the license
5. Wait for approval — usually instant, occasionally a few minutes

### Step 2: Create a Hugging Face access token

1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"** → Type: "Read" → give it any name → Generate
3. Copy the token (starts with `hf_...`)

### Step 3: Log in from your terminal

```bash
huggingface-cli login
```
Paste your token when prompted. (It's stored in `~/.cache/huggingface/token`.)

### Step 4: Hardware check

Llama-3-8B in **4-bit quantization** needs about **6 GB of GPU VRAM**. Check yours:

```bash
nvidia-smi
```

You're looking for the **"Memory-Usage"** column. You need ~6 GB free for 4-bit, ~16 GB for fp16.

### Step 5: Install 4-bit support

```bash
pip install bitsandbytes peft
```

### Step 6: Run

```bash
python -m src.task4.run
```

The first run downloads the model (~16 GB) into `~/.cache/huggingface/` — this happens once. Inference for 200 test samples takes 20–40 minutes on a mid-range GPU.

## Option B — Run on Google Colab (no GPU needed locally)

Google Colab gives you a free T4 GPU (16 GB) which is enough.

1. Upload the project to Colab (or clone it from GitHub inside Colab):
   ```python
   !git clone https://github.com/<your-username>/nalapro-project.git
   %cd nalapro-project
   !pip install -r requirements.txt
   !pip install bitsandbytes peft
   ```
2. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**
3. Log in:
   ```python
   from huggingface_hub import login
   login(token="hf_your_token_here")

   import wandb
   wandb.login(key="your_wandb_key")
   ```
4. Run:
   ```python
   !python -m src.task4.run
   ```

## Option C — Use a smaller open model (no Llama license needed)

If you can't or don't want to wait for Meta's Llama-3 approval, swap in a smaller, ungated instruct model. Edit `config.py`:

```python
# Pick ONE of these (smallest to largest)
LLAMA_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"      # ~3 GB VRAM, decent quality
LLAMA_MODEL = "google/gemma-2-2b-it"            # ~5 GB VRAM
LLAMA_MODEL = "microsoft/Phi-3-mini-4k-instruct" # ~8 GB VRAM, strong
LLAMA_MODEL = "Qwen/Qwen2.5-7B-Instruct"        # ~16 GB VRAM, very strong
```

None of these require license acceptance. **For your report**, just mention that you used `<model>` as a Llama-3 substitute due to resource/access constraints. This is normal and acceptable.

You can also run these on CPU (slow but works) by setting `LLAMA_USE_4BIT = False`.

## Option D — Use the Llama-3 via an API (no GPU required)

If you want the real Llama-3 but have no GPU, use a hosted API. Two free options:

### Groq (free tier, fast)
1. Sign up at https://console.groq.com (free)
2. Get an API key
3. Install: `pip install groq`
4. Replace `src/task4/llama.py:load_llama()` and `run_llama_experiment()` to use the Groq client.

### Hugging Face Inference API (free tier, rate-limited)
1. Use your HF token
2. Use the InferenceClient:
   ```python
   from huggingface_hub import InferenceClient
   client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token="hf_...")
   response = client.text_generation(prompt, max_new_tokens=30)
   ```

For a student project, **Options A, B, or C are all fully acceptable**. Document whichever you chose in your report's "Tools" section.

## What to discuss in the report

The task description says: "Evaluate and discuss the results." Things to write about:

1. **Zero-shot vs few-shot performance** — does adding 3 examples help? By how much?
2. **Comparison to BERT** (Task 2) — does a frozen 8B-param LLM beat a fine-tuned 110M BERT? (Spoiler: usually no, on this dataset.)
3. **Where the LLM fails** — look at the confusion matrix. Are confusions between semantically close classes (e.g. `talk.politics.misc` vs `talk.politics.guns`)?
4. **Unparseable rate** — the script prints what fraction of responses couldn't be parsed. Discuss the failure modes.
5. **Cost / efficiency** — fine-tuning BERT was a one-time cost (~30 min). Inference is fast. Llama-3 requires no training but each prediction is slow. Trade-offs.

## Common errors

| Error | Fix |
|-------|-----|
| `OSError: You are trying to access a gated repo` | You haven't accepted Llama-3's license, or `huggingface-cli login` failed. Re-do steps 1-3. |
| `bitsandbytes` install fails | Skip it — set `LLAMA_USE_4BIT = False` and use Option C with a smaller model, or use Colab. |
| `CUDA out of memory` | Reduce `LLAMA_EVAL_SAMPLES` to 50, or switch to a smaller model in `config.py`. |
| Very slow inference (>1 sec/sample) | This is normal on CPU. Either accept it or use Colab/GPU. |
| Many unparseable responses | The model is being chatty. Lower `LLAMA_MAX_NEW_TOKENS` to 10, or improve the prompt to be more direct. |
