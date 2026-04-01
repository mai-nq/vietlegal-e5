# VietLegal-E5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 7-stage pipeline to fine-tune `intfloat/multilingual-e5-large` for Vietnamese legal retrieval, targeting NDCG@10 > 0.8976 on Zalo Legal benchmark.

**Architecture:** Standalone Python scripts per pipeline stage, unified by a single `config.yaml` and shared `scripts/utils.py`. Each stage reads config, performs its work, writes outputs to `data/` or `models/`. W&B tracks all experiments. Multi-GPU via `accelerate launch`.

**Tech Stack:** sentence-transformers>=3.0, transformers, datasets, torch, accelerate, faiss-gpu, wandb, vllm, openai, optimum, rank_bm25

**Spec:** `docs/superpowers/specs/2026-03-24-vietlegal-e5-design.md`

---

## File Map

| File | Responsibility | Created in Task |
|------|---------------|-----------------|
| `config.yaml` | All hyperparams, paths, flags | 1 |
| `requirements.txt` | Pinned dependencies | 1 |
| `.gitignore` | Exclude data/models/eval from git | 1 |
| `scripts/__init__.py` | Make scripts dir importable as package | 1 |
| `scripts/utils.py` | Config loading, prefix helpers, W&B init, logging | 2 |
| `scripts/prepare_data.py` | Stage 0: load legal docs, chunk, filter, save parquet | 3 |
| `scripts/train_tsdae.py` | Stage 1: TSDAE domain adaptation | 4 |
| `scripts/prepare_training.py` | Stage 2.1: load query-passage pairs, prefix, split | 5 |
| `scripts/generate_queries.py` | Stage 2.4: vLLM synthetic query generation | 6 |
| `scripts/train_contrastive.py` | Stage 3: contrastive fine-tuning R1 | 7 |
| `scripts/mine_hard_negatives.py` | Stage 4.1-4.2: encode corpus, FAISS index, mine + cross-encoder filter | 8 |
| `scripts/train_hard_neg.py` | Stage 4.3: contrastive R2 with hard negatives | 9 |
| `scripts/train_multitask.py` | Stage 5: multi-task blending | 10 |
| `scripts/evaluate.py` | Stage 6: eval all checkpoints + baselines on Zalo Legal | 11 |
| `scripts/export.py` | Stage 7: HF Hub push, ONNX export | 12 |

---

### Task 1: Project Scaffolding — config.yaml + requirements.txt + setup

**Files:**
- Create: `config.yaml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `scripts/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p data models eval scripts
```

- [ ] **Step 2: Write `requirements.txt`**

```txt
sentence-transformers>=3.0,<4.0
transformers>=4.40,<5.0
datasets>=2.18,<3.0
torch>=2.1,<3.0
accelerate>=0.28,<1.0
faiss-gpu>=1.7,<2.0
wandb>=0.16,<1.0
pandas>=2.0,<3.0
pyarrow>=14.0,<18.0
pyyaml>=6.0,<7.0
rank_bm25>=0.2.2,<1.0
optimum[onnxruntime]>=1.17,<2.0
huggingface_hub>=0.21,<1.0
openai>=1.12,<2.0
vllm>=0.4,<1.0
scikit-learn>=1.3,<2.0
```

- [ ] **Step 3: Write `.gitignore`**

```
# Data artifacts (large binary files)
data/*.parquet
data/synth_checkpoint.json

# Model checkpoints (multi-GB)
models/

# Eval results (regenerable)
eval/

# Python
__pycache__/
*.pyc
.eggs/

# IDE
.vscode/
.idea/

# W&B
wandb/
```

- [ ] **Step 4: Create `scripts/__init__.py`**

Create an empty `scripts/__init__.py` so the `scripts` directory is importable as a Python package (needed for `from scripts.utils import ...`).

- [ ] **Step 5: Write `config.yaml`**

Write the full config from the spec (see `docs/superpowers/specs/2026-03-24-vietlegal-e5-design.md`, lines 60-171). Copy it exactly — all sections: `paths`, `data`, `tsdae`, `contrastive_r1`, `hard_neg`, `contrastive_r2`, `multitask`, `synth`, `eval`, `export`, `wandb`, `hardware`, `split`.

- [ ] **Step 6: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 7: Authenticate services**

```bash
# W&B login (required for experiment tracking in all training scripts)
wandb login

# HuggingFace login (required for dataset downloads and model push in Stage 7)
huggingface-cli login
```

- [ ] **Step 8: Verify config loads**

```bash
python -c "import yaml; cfg = yaml.safe_load(open('config.yaml')); print(list(cfg.keys()))"
```

Expected output: `['paths', 'data', 'tsdae', 'contrastive_r1', 'hard_neg', 'contrastive_r2', 'multitask', 'synth', 'eval', 'export', 'wandb', 'hardware', 'split']`

- [ ] **Step 9: Commit**

```bash
git init
git add config.yaml requirements.txt .gitignore scripts/__init__.py
git commit -m "chore: scaffold project with config, deps, gitignore, and package init"
```

---

### Task 2: Shared Utilities — `scripts/utils.py`

**Files:**
- Create: `scripts/utils.py`

- [ ] **Step 1: Write `scripts/utils.py`**

This file provides 4 things used by every other script:

```python
"""Shared utilities for VietLegal-E5 pipeline."""

import logging
import os
from pathlib import Path

import yaml
import wandb


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the YAML config as a dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve paths relative to base_dir
    base = Path(cfg["paths"]["base_dir"])
    cfg["paths"]["data_dir"] = str(base / cfg["paths"]["data_dir"])
    cfg["paths"]["model_dir"] = str(base / cfg["paths"]["model_dir"])
    cfg["paths"]["eval_dir"] = str(base / cfg["paths"]["eval_dir"])
    return cfg


def add_prefix(text: str, prefix: str) -> str:
    """Add 'query: ' or 'passage: ' prefix if not already present."""
    if text.startswith(prefix):
        return text
    return prefix + text


def add_query_prefix(text: str) -> str:
    return add_prefix(text, "query: ")


def add_passage_prefix(text: str) -> str:
    return add_prefix(text, "passage: ")


def init_wandb(cfg: dict, run_name: str):
    """Initialize W&B run from config."""
    wandb.init(
        project=cfg["wandb"]["project"],
        name=run_name,
        tags=cfg["wandb"]["tags"],
        config=cfg,
    )


def setup_logging(name: str) -> logging.Logger:
    """Set up a logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger


def ensure_dirs(cfg: dict):
    """Create output directories if they don't exist."""
    for key in ["data_dir", "model_dir", "eval_dir"]:
        os.makedirs(cfg["paths"][key], exist_ok=True)
```

- [ ] **Step 2: Verify import works**

```bash
cd /home/mainq/vietlegal_e5 && python -c "from scripts.utils import load_config, add_query_prefix, add_passage_prefix; cfg = load_config(); print('OK:', add_query_prefix('test'), add_passage_prefix('test'))"
```

Expected: `OK: query: test passage: test`

- [ ] **Step 3: Commit**

```bash
git add scripts/utils.py
git commit -m "feat: add shared utils — config loader, prefix helpers, W&B init, logging"
```

---

### Task 3: Stage 0 — Data Preparation (`scripts/prepare_data.py`)

**Files:**
- Create: `scripts/prepare_data.py`

This is the most complex data processing script. It loads 518K legal docs, chunks them into ~2-5M chunks, and saves to parquet.

- [ ] **Step 1: Write `scripts/prepare_data.py`**

The script must implement:

1. **Load** both `metadata` and `content` configs from `th1nhng0/vietnamese-legal-documents` via `datasets.load_dataset()`. Join on `id` column using pandas merge.

2. **Filter**: remove docs with `len(content) < 100` chars. Deduplicate by `document_number` (keep first).

3. **Tokenizer**: load `xlm-roberta-large` tokenizer from `transformers` for token counting.

4. **Chunking function** `chunk_document(text, tokenizer, cfg) -> list[str]`:
   - Try regex split using capturing group: `parts = re.split(r'(Điều\s+\d+[^.]*\.)', text)`
   - `re.split` with a capturing group returns alternating: `[before_match, match1, after_match1, match2, after_match2, ...]`. Recombine header+body pairs:
     ```python
     chunks = []
     i = 0
     if parts[0].strip():  # text before first Điều
         chunks.append(parts[0].strip())
         i = 1
     else:
         i = 1
     while i < len(parts) - 1:
         header = parts[i]        # "Điều 1. ..."
         body = parts[i + 1]      # text until next Điều
         chunks.append((header + body).strip())
         i += 2
     if i < len(parts) and parts[i].strip():
         chunks.append(parts[i].strip())
     ```
   - For each regex chunk: if `len(tokenizer.encode(chunk)) > cfg["data"]["chunk_max_tokens"]` → apply sliding window to that chunk
   - If no regex matches at all (only 1 part returned) → apply sliding window to entire text
   - Sliding window: encode text to token IDs, slice windows of `chunk_window_tokens` with `chunk_overlap_tokens` overlap, decode each window back to text
   - Merge: after all chunking, merge consecutive chunks where `len(tokenizer.encode(chunk)) < cfg["data"]["chunk_min_tokens"]` by concatenating with the next chunk

5. **Main loop**: iterate docs, chunk each, attach metadata (`legal_type`, `legal_sectors`, `source_doc_id`, `chunk_index`), assign `chunk_id = f"{source_doc_id}_{chunk_index}"`

6. **Save**: `pd.DataFrame(all_chunks).to_parquet(data_dir / "chunks.parquet", index=False)`

7. Log stats: total docs loaded, docs after filter, total chunks created.

Key imports: `datasets`, `transformers.AutoTokenizer`, `pandas`, `re`, `pathlib.Path`

Entry point: `if __name__ == "__main__"` block that calls `load_config()`, `setup_logging()`, `ensure_dirs()`, then runs the pipeline.

- [ ] **Step 2: Dry-run test with small subset**

```bash
cd /home/mainq/vietlegal_e5 && python scripts/prepare_data.py 2>&1 | head -50
```

Verify: script starts downloading dataset, logs progress. If dataset is large, can `Ctrl+C` after seeing the first log lines confirming data loading works. The full run will produce `data/chunks.parquet`.

- [ ] **Step 3: Validate output**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/chunks.parquet')
print(f'Chunks: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print(f'Sample text length: {df[\"text\"].str.len().describe()}')
print(df.head(2)[['chunk_id','text']].to_string())
"
```

Expected: ~2-5M rows, columns `['chunk_id', 'text', 'source_doc_id', 'legal_type', 'legal_sectors', 'chunk_index']`

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare_data.py
git commit -m "feat: Stage 0 — data preparation, legal doc chunking pipeline"
```

---

### Task 4: Stage 1 — TSDAE Pre-training (`scripts/train_tsdae.py`)

**Files:**
- Create: `scripts/train_tsdae.py`

- [ ] **Step 1: Write `scripts/train_tsdae.py`**

The script implements TSDAE domain adaptation using ST v3 API:

1. Load config, init W&B with run_name `"tsdae"`, setup logging.

2. Load `intfloat/multilingual-e5-large` as `SentenceTransformer`. Set `max_seq_length`:
   ```python
   model = SentenceTransformer(cfg["tsdae"]["base_model"])
   model.max_seq_length = cfg["tsdae"]["max_seq_len"]  # 512
   ```

3. Load chunks from `data/chunks.parquet`, create HuggingFace `Dataset` with `sentence_0` column (the original text). `DenoisingAutoEncoderLoss` handles corruption internally — do NOT apply manual noise:
   ```python
   from datasets import Dataset
   df = pd.read_parquet(data_dir / "chunks.parquet")
   dataset = Dataset.from_dict({"sentence_0": df["text"].tolist()})
   ```

   **IMPORTANT**: Do NOT add a manual denoising transform via `set_transform()`. The `DenoisingAutoEncoderLoss` corrupts the input internally (deletes ~60% of tokens). Adding manual corruption would cause double-corruption and severely degrade training. The dataset must provide clean text only.

4. Create `DenoisingAutoEncoderLoss` with `tie_encoder_decoder=True` (uses the XLM-R backbone as both encoder and decoder — no separate decoder model needed):
   ```python
   from sentence_transformers.losses import DenoisingAutoEncoderLoss
   loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)
   ```

5. Configure `SentenceTransformerTrainingArguments`:
   ```python
   from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer

   args = SentenceTransformerTrainingArguments(
       output_dir=cfg["tsdae"]["output_dir"],
       num_train_epochs=cfg["tsdae"]["epochs"],
       per_device_train_batch_size=cfg["tsdae"]["batch_size"],
       learning_rate=cfg["tsdae"]["lr"],
       lr_scheduler_type="linear",  # matches config's "warmup_linear" — linear decay after warmup
       warmup_steps=cfg["tsdae"]["warmup_steps"],
       weight_decay=cfg["tsdae"]["weight_decay"],
       bf16=cfg["hardware"]["bf16"],
       dataloader_num_workers=cfg["hardware"]["dataloader_workers"],
       save_total_limit=2,
       logging_steps=100,
       seed=cfg["hardware"]["seed"],
       report_to="wandb",
   )
   ```

   **Note on `resume_from_checkpoint`**: Do not set `resume_from_checkpoint=True` in args. Instead, call `trainer.train(resume_from_checkpoint=True)` if you want to resume, or `trainer.train()` for a fresh start. The Trainer gracefully handles the case where no checkpoint exists when called this way.

6. Create trainer and train:
   ```python
   trainer = SentenceTransformerTrainer(
       model=model,
       args=args,
       train_dataset=dataset,
       loss=loss,
   )
   trainer.train()
   model.save(cfg["tsdae"]["output_dir"])
   ```

7. Launch with: `accelerate launch --num_processes=2 scripts/train_tsdae.py`

- [ ] **Step 2: Verify script starts training**

```bash
cd /home/mainq/vietlegal_e5 && accelerate launch --num_processes=2 scripts/train_tsdae.py 2>&1 | head -30
```

Verify: model loads, training starts, W&B run appears. Full training takes ~6-8 hours.

- [ ] **Step 3: After training completes — validate output**

```bash
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('models/vietlegal-e5-tsdae')
print(f'Model loaded. Dim: {m.get_sentence_embedding_dimension()}')
emb = m.encode(['passage: Điều 1. Phạm vi điều chỉnh'])
print(f'Embedding shape: {emb.shape}')
"
```

Expected: `Dim: 1024`, `Embedding shape: (1, 1024)`

- [ ] **Step 4: Commit**

```bash
git add scripts/train_tsdae.py
git commit -m "feat: Stage 1 — TSDAE domain-adaptive pre-training script"
```

---

### Task 5: Stage 2.1 — Training Data Preparation (`scripts/prepare_training.py`)

**Files:**
- Create: `scripts/prepare_training.py`

- [ ] **Step 1: Write `scripts/prepare_training.py`**

Simple script:

1. Load config, setup logging, ensure dirs.

2. Load dataset:
   ```python
   from datasets import load_dataset
   ds = load_dataset(cfg["data"]["query_pairs_dataset"], split="train")
   df = ds.to_pandas()
   ```

3. Add prefixes:
   ```python
   from scripts.utils import add_query_prefix, add_passage_prefix
   df["query"] = df["query"].apply(add_query_prefix)
   df["positive"] = df["positive"].apply(add_passage_prefix)
   ```

4. Split:
   ```python
   from sklearn.model_selection import train_test_split
   # First split: 95% train, 5% temp
   train_df, temp_df = train_test_split(df, test_size=0.05, random_state=cfg["split"]["seed"])
   # Second split: 50/50 of temp -> 2.5% val, 2.5% test
   val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=cfg["split"]["seed"])
   ```

5. Save:
   ```python
   data_dir = Path(cfg["paths"]["data_dir"])
   train_df.to_parquet(data_dir / "train.parquet", index=False)
   val_df.to_parquet(data_dir / "val.parquet", index=False)
   test_df.to_parquet(data_dir / "test.parquet", index=False)
   ```

6. Log counts: `logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")`

Add `scikit-learn` to `requirements.txt` for `train_test_split`.

- [ ] **Step 2: Run script**

```bash
cd /home/mainq/vietlegal_e5 && python scripts/prepare_training.py
```

- [ ] **Step 3: Validate output**

```bash
python -c "
import pandas as pd
for name in ['train', 'val', 'test']:
    df = pd.read_parquet(f'data/{name}.parquet')
    print(f'{name}: {len(df):,} rows')
    print(f'  query prefix check: {df[\"query\"].iloc[0][:10]}')
    print(f'  passage prefix check: {df[\"positive\"].iloc[0][:12]}')
"
```

Expected: ~481K train, ~12.5K val, ~12.5K test. All queries start with `"query: "`, all passages start with `"passage: "`.

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare_training.py requirements.txt
git commit -m "feat: Stage 2.1 — prepare training data with prefix and split"
```

---

### Task 6: Stage 2.4 — Synthetic Query Generation (`scripts/generate_queries.py`)

**Files:**
- Create: `scripts/generate_queries.py`

- [ ] **Step 1: Write `scripts/generate_queries.py`**

This is the most complex infrastructure script. It must:

1. **Load config, chunks**: Read `data/chunks.parquet`. Load checkpoint if exists (`data/synth_checkpoint.json` — stores last processed chunk index).

2. **Launch vLLM server** as subprocess:
   ```python
   import subprocess, time, requests

   cmd = [
       "python", "-m", "vllm.entrypoints.openai.api_server",
       "--model", cfg["synth"]["model_name"],
       "--tensor-parallel-size", str(cfg["synth"]["vllm_tensor_parallel"]),
       "--max-model-len", str(cfg["synth"]["vllm_max_model_len"]),
       "--port", str(cfg["synth"]["vllm_port"]),
       "--dtype", "auto",
   ]
   server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   ```

3. **Health check loop with auto-fallback**:
   ```python
   import requests, time

   port = cfg["synth"]["vllm_port"]
   healthy = False
   for _ in range(60):  # 10 minutes max
       try:
           r = requests.get(f"http://localhost:{port}/health", timeout=5)
           if r.status_code == 200:
               healthy = True
               break
       except requests.ConnectionError:
           pass
       time.sleep(10)

   if not healthy:
       logger.warning("Primary model failed to load. Trying fallback FP8 model...")
       server_proc.terminate()
       server_proc.wait()
       cmd[cmd.index(cfg["synth"]["model_name"])] = cfg["synth"]["fallback_model"]
       server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       # Retry health check loop for fallback
       for _ in range(60):
           try:
               r = requests.get(f"http://localhost:{port}/health", timeout=5)
               if r.status_code == 200:
                   healthy = True
                   break
           except requests.ConnectionError:
               pass
           time.sleep(10)
       if not healthy:
           raise RuntimeError("Both primary and fallback models failed to load in vLLM")
   ```

4. **Async batch generation**: Use `openai.AsyncOpenAI` (base_url=`http://localhost:{port}/v1`) with `asyncio.gather` for concurrent requests. Process chunks in batches of `cfg["synth"]["batch_size"]`. Pass generation params from config:

   ```python
   import asyncio
   from openai import AsyncOpenAI

   client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")

   async def generate_for_chunk(chunk_text: str) -> list[str]:
       response = await client.chat.completions.create(
           model=cfg["synth"]["model_name"],
           messages=[{"role": "user", "content": prompt.format(chunk_text=chunk_text)}],
           max_tokens=cfg["synth"]["max_new_tokens"],   # 512
           temperature=cfg["synth"]["temperature"],       # 0.7
       )
       return response.choices[0].message.content
   ```

   Prompt per chunk:
   ```
   Bạn là chuyên gia pháp luật Việt Nam. Dựa trên đoạn văn bản pháp luật sau,
   hãy viết đúng 3 câu hỏi pháp lý mà đoạn văn này có thể trả lời.
   Câu hỏi phải tự nhiên, đa dạng về cách hỏi.
   Mỗi câu hỏi trên một dòng, đánh số 1., 2., 3.

   Đoạn văn: {chunk_text}
   ```

5. **Parse responses**: Split by `\n`, extract lines starting with `1.`, `2.`, `3.`. Each becomes a (query, passage) pair. Apply prefixes.

6. **Checkpoint**: Every `cfg["synth"]["checkpoint_every_n"]` chunks, save progress to `data/synth_checkpoint.json` and flush accumulated pairs to `data/synthetic_queries.parquet` (append mode).

7. **Merge into train only**:
   ```python
   synth_df = pd.read_parquet(data_dir / "synthetic_queries.parquet")
   train_df = pd.read_parquet(data_dir / "train.parquet")
   merged = pd.concat([train_df, synth_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
   merged.to_parquet(data_dir / "train.parquet", index=False)
   # val.parquet and test.parquet are NOT touched
   ```

8. **Cleanup**: Kill vLLM server process, remove checkpoint file.

- [ ] **Step 2: Test vLLM server launch (manual check)**

```bash
cd /home/mainq/vietlegal_e5 && python scripts/generate_queries.py 2>&1 | head -30
```

Verify: vLLM starts loading model, health check polling begins. If Qwen3-80B doesn't fit, script should auto-fallback to FP8 variant.

- [ ] **Step 3: After generation completes — validate output**

```bash
python -c "
import pandas as pd
synth = pd.read_parquet('data/synthetic_queries.parquet')
print(f'Synthetic pairs: {len(synth):,}')
print(f'Sample query: {synth[\"query\"].iloc[0][:80]}')
train = pd.read_parquet('data/train.parquet')
print(f'Updated train: {len(train):,}')
"
```

Expected: ~500K+ synthetic pairs, updated train ~1M+.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_queries.py
git commit -m "feat: Stage 2.4 — synthetic query generation via vLLM + Qwen3"
```

---

### Task 7: Stage 3 — Contrastive Fine-tuning R1 (`scripts/train_contrastive.py`)

**Files:**
- Create: `scripts/train_contrastive.py`

- [ ] **Step 1: Write `scripts/train_contrastive.py`**

1. Load config, init W&B run `"contrastive-r1"`.

2. **Resolve base model**: If `models/vietlegal-e5-tsdae/` exists, use it. Otherwise fall back to `intfloat/multilingual-e5-large`. Log which is used.
   ```python
   tsdae_path = Path(cfg["tsdae"]["output_dir"])
   base_model = str(tsdae_path) if tsdae_path.exists() else cfg["tsdae"]["base_model"]
   model = SentenceTransformer(base_model)
   ```

3. **Load datasets**: Read `data/train.parquet` and `data/val.parquet`. Convert to HuggingFace `Dataset` with columns `sentence_0` (query) and `sentence_1` (positive):
   ```python
   from datasets import Dataset
   train_df = pd.read_parquet(data_dir / "train.parquet")
   train_ds = Dataset.from_dict({"sentence_0": train_df["query"].tolist(), "sentence_1": train_df["positive"].tolist()})
   # Same for val
   ```

4. **Loss**:
   ```python
   from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
   mnrl = MultipleNegativesRankingLoss(model)
   loss = MatryoshkaLoss(model, mnrl, matryoshka_dims=cfg["contrastive_r1"]["matryoshka_dims"])
   ```

5. **Evaluator**: Build `InformationRetrievalEvaluator` from val set with distractor corpus. The corpus must be larger than the query set — using only val positive passages (1:1 mapping) produces artificially inflated metrics. Add distractors from the training set:
   ```python
   from sentence_transformers.evaluation import InformationRetrievalEvaluator

   # Queries from val set
   queries = {f"q{i}": row["query"] for i, row in val_df.iterrows()}

   # Corpus = val positives + random sample of train positives as distractors
   corpus = {}
   relevant_docs = {}
   for i, row in val_df.iterrows():
       corpus[f"c{i}"] = row["positive"]
       relevant_docs[f"q{i}"] = {f"c{i}": 1}

   # Add ~10K distractors from train (not matching any val query)
   train_sample = train_df.sample(n=min(10000, len(train_df)), random_state=42)
   for j, row in enumerate(train_sample.itertuples()):
       corpus[f"d{j}"] = row.positive

   evaluator = InformationRetrievalEvaluator(
       queries=queries, corpus=corpus, relevant_docs=relevant_docs,
       name="val", ndcg_at_k=[10], show_progress_bar=True,
   )
   ```

   This gives a corpus of ~22.5K (12.5K val + 10K distractors) with 12.5K queries, providing meaningful NDCG@10 scores.

6. **Training args**:
   ```python
   args = SentenceTransformerTrainingArguments(
       output_dir=cfg["contrastive_r1"]["output_dir"],
       num_train_epochs=cfg["contrastive_r1"]["epochs"],
       per_device_train_batch_size=cfg["contrastive_r1"]["per_device_batch_size"],
       gradient_accumulation_steps=cfg["contrastive_r1"]["grad_accum_steps"],
       learning_rate=cfg["contrastive_r1"]["lr"],
       warmup_ratio=cfg["contrastive_r1"]["warmup_ratio"],
       lr_scheduler_type=cfg["contrastive_r1"]["scheduler"],
       bf16=cfg["hardware"]["bf16"],
       eval_steps=cfg["contrastive_r1"]["eval_steps"],
       eval_strategy="steps",
       save_total_limit=cfg["contrastive_r1"]["save_total_limit"],
       load_best_model_at_end=True,
       metric_for_best_model="val_ndcg@10",
       dataloader_num_workers=cfg["hardware"]["dataloader_workers"],
       seed=cfg["hardware"]["seed"],
       report_to="wandb",
       resume_from_checkpoint=True,
   )
   ```

7. **Train**:
   ```python
   trainer = SentenceTransformerTrainer(
       model=model, args=args, train_dataset=train_ds,
       eval_dataset=val_ds, loss=loss, evaluator=evaluator,
   )
   trainer.train()
   model.save(cfg["contrastive_r1"]["output_dir"])
   ```

8. Launch: `accelerate launch --num_processes=2 scripts/train_contrastive.py`

- [ ] **Step 2: Verify script starts training**

```bash
cd /home/mainq/vietlegal_e5 && accelerate launch --num_processes=2 scripts/train_contrastive.py 2>&1 | head -30
```

- [ ] **Step 3: After training — validate output**

```bash
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('models/vietlegal-e5-r1')
print(f'R1 model loaded. Dim: {m.get_sentence_embedding_dimension()}')
emb = m.encode(['query: quyền sở hữu trí tuệ là gì'])
print(f'Embedding shape: {emb.shape}')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_contrastive.py
git commit -m "feat: Stage 3 — contrastive fine-tuning R1 with Matryoshka loss"
```

---

### Task 8: Stage 4.1-4.2 — Hard Negative Mining (`scripts/mine_hard_negatives.py`)

**Files:**
- Create: `scripts/mine_hard_negatives.py`

- [ ] **Step 1: Write `scripts/mine_hard_negatives.py`**

1. Load config, R1 model, training data.

2. **Collect all unique passages** from `data/train.parquet`:
   ```python
   train_df = pd.read_parquet(data_dir / "train.parquet")
   passages = train_df["positive"].unique().tolist()
   queries = train_df["query"].tolist()
   query_to_positive = dict(zip(train_df["query"], train_df["positive"]))
   ```

3. **Encode passages**:
   ```python
   model = SentenceTransformer(cfg["contrastive_r1"]["output_dir"])
   passage_embeddings = model.encode(
       passages, batch_size=cfg["hard_neg"]["encode_batch_size"],
       show_progress_bar=True, normalize_embeddings=True,
   )
   ```

4. **Build FAISS index**:
   ```python
   import faiss
   import numpy as np
   dim = passage_embeddings.shape[1]
   if cfg["hard_neg"]["faiss_index_type"] == "ivf" and len(passages) > 5_000_000:
       quantizer = faiss.IndexFlatIP(dim)
       index = faiss.IndexIVFFlat(quantizer, dim, cfg["hard_neg"]["faiss_ivf_nlist"])
       index.train(passage_embeddings)
   else:
       index = faiss.IndexFlatIP(dim)
   index.add(passage_embeddings)
   ```

5. **Mine per query**: Encode queries in batches, search top_k=100. For each query:
   - Get retrieved passage indices
   - Remove true positive (match by text)
   - Take indices from `neg_range_start` to `neg_range_end` as candidates

6. **Cross-encoder filtering** (if `cfg["hard_neg"]["use_cross_encoder"]`):
   ```python
   from sentence_transformers import CrossEncoder
   reranker = CrossEncoder(cfg["hard_neg"]["cross_encoder_model"])

   # For each query's candidate hard negatives:
   # Strip prefixes before passing to cross-encoder (rerankers expect raw text)
   raw_query = query.replace("query: ", "", 1)
   raw_candidates = [c.replace("passage: ", "", 1) for c in candidates]

   # Score all (query, candidate) pairs
   pairs = [[raw_query, c] for c in raw_candidates]
   scores = reranker.predict(pairs, batch_size=256)

   # REMOVE candidates with score > threshold — these are likely TRUE positives
   # (false negatives in our negative candidate set) and would hurt training
   # KEEP candidates with score <= threshold — these are genuine hard negatives
   filtered = [
       (cand, score) for cand, score in zip(candidates, scores)
       if score <= cfg["hard_neg"]["cross_encoder_threshold"]
   ]
   # Take top max_negatives_per_query by descending score (hardest remaining negatives)
   filtered.sort(key=lambda x: x[1], reverse=True)
   hard_negs = [cand for cand, _ in filtered[:cfg["hard_neg"]["max_negatives_per_query"]]]
   ```

   Process all queries in batches for efficiency.

7. **Save**: Build DataFrame with columns `query`, `positive`, `hard_negative`. Save to `data/hard_negatives.parquet`.

- [ ] **Step 2: Run mining**

```bash
cd /home/mainq/vietlegal_e5 && python scripts/mine_hard_negatives.py 2>&1 | tail -20
```

- [ ] **Step 3: Validate output**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/hard_negatives.parquet')
print(f'Hard negative triplets: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print(f'Sample:')
print(df.iloc[0].to_string()[:200])
"
```

Expected: columns `['query', 'positive', 'hard_negative']`, hundreds of thousands of triplets.

- [ ] **Step 4: Commit**

```bash
git add scripts/mine_hard_negatives.py
git commit -m "feat: Stage 4.1-4.2 — hard negative mining with FAISS + cross-encoder filter"
```

---

### Task 9: Stage 4.3 — Contrastive R2 (`scripts/train_hard_neg.py`)

**Files:**
- Create: `scripts/train_hard_neg.py`

- [ ] **Step 1: Write `scripts/train_hard_neg.py`**

Very similar to `train_contrastive.py` with these differences:

- Base model: `models/vietlegal-e5-r1/`
- Data: triplets from `data/hard_negatives.parquet` — columns map to `sentence_0` (query), `sentence_1` (positive), `sentence_2` (hard_negative)
- Same loss: `MatryoshkaLoss(MultipleNegativesRankingLoss)` — MNRL with 3 columns automatically treats column 2 as a hard negative
- Same evaluator setup as R1 (from val.parquet with distractors)
- Output: `models/vietlegal-e5-r2/`

**Config inheritance**: R2 has its own `lr` and `epochs` but reads batch/dim params from R1:
```python
args = SentenceTransformerTrainingArguments(
    output_dir=cfg["contrastive_r2"]["output_dir"],
    num_train_epochs=cfg["contrastive_r2"]["epochs"],           # 2 (R2-specific)
    learning_rate=cfg["contrastive_r2"]["lr"],                   # 5e-6 (R2-specific)
    per_device_train_batch_size=cfg["contrastive_r1"]["per_device_batch_size"],  # 32 (from R1)
    gradient_accumulation_steps=cfg["contrastive_r1"]["grad_accum_steps"],       # 2 (from R1)
    # ... other args same as R1
)
dims = cfg["contrastive_r1"]["matryoshka_dims"]  # [1024, 512, 256, 128] (from R1)
```

- [ ] **Step 2: Launch training**

```bash
cd /home/mainq/vietlegal_e5 && accelerate launch --num_processes=2 scripts/train_hard_neg.py 2>&1 | head -30
```

- [ ] **Step 3: After training — validate**

```bash
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('models/vietlegal-e5-r2')
print(f'R2 model loaded. Dim: {m.get_sentence_embedding_dimension()}')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_hard_neg.py
git commit -m "feat: Stage 4.3 — contrastive R2 training with hard negatives"
```

---

### Task 10: Stage 5 — Multi-task Blending (`scripts/train_multitask.py`)

**Files:**
- Create: `scripts/train_multitask.py`

- [ ] **Step 1: Write `scripts/train_multitask.py`**

1. Load config, R2 model, init W&B run `"multitask"`.

2. **Build retrieval dataset**: Same as R1 — `sentence_0` (query), `sentence_1` (positive) from `data/train.parquet`.

3. **Build classification dataset** from `data/chunks.parquet`:
   ```python
   chunks_df = pd.read_parquet(data_dir / "chunks.parquet")
   # Group chunks by legal_type
   # For each legal_type with >= 2 chunks, sample pairs of chunks as (sentence_0, sentence_1)
   # Both get "passage: " prefix since they're passage-passage similarity
   classification_pairs = []
   for legal_type, group in chunks_df.groupby("legal_type"):
       texts = group["text"].tolist()
       if len(texts) < 2:
           continue
       # Sample pairs — take consecutive pairs to limit combinatorial explosion
       for i in range(0, len(texts) - 1, 2):
           classification_pairs.append({
               "sentence_0": add_passage_prefix(texts[i]),
               "sentence_1": add_passage_prefix(texts[i + 1]),
           })
   classification_df = pd.DataFrame(classification_pairs)
   ```

4. **Build STS dataset**: Try loading `stsb_multi_mt` Vietnamese split:
   ```python
   try:
       sts_ds = load_dataset("stsb_multi_mt", "vi", split="train")
       sts_df = pd.DataFrame({
           "sentence_0": [add_passage_prefix(r["sentence1"]) for r in sts_ds],
           "sentence_1": [add_passage_prefix(r["sentence2"]) for r in sts_ds],
           "score": [r["similarity_score"] / 5.0 for r in sts_ds],  # normalize to 0-1
       })
       has_sts = True
   except Exception:
       has_sts = False
   ```

5. **Adjust dataset sizes for sampling ratios**: Calculate target sizes based on retrieval dataset size and 70/20/10 ratios. Oversample/undersample classification and STS datasets accordingly:
   ```python
   retrieval_size = len(retrieval_df)
   target_total = int(retrieval_size / cfg["multitask"]["sampling_ratios"]["retrieval"])
   target_class = int(target_total * cfg["multitask"]["sampling_ratios"]["classification"])
   # Oversample/undersample classification_df to target_class rows
   ```
   If no STS, redistribute: 80% retrieval, 20% classification.

6. **Create dataset and loss dicts**:
   ```python
   from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss, CoSENTLoss

   train_datasets = {
       "retrieval": retrieval_ds,
       "classification": classification_ds,
   }
   losses = {
       "retrieval": MatryoshkaLoss(model, MultipleNegativesRankingLoss(model), matryoshka_dims=dims),
       "classification": MatryoshkaLoss(model, MultipleNegativesRankingLoss(model), matryoshka_dims=dims),
   }
   if has_sts:
       train_datasets["sts"] = sts_ds
       losses["sts"] = MatryoshkaLoss(model, CoSENTLoss(model), matryoshka_dims=dims)
   ```

7. **Training args**: lr=5e-6, epochs=1, `multi_dataset_batch_sampler="PROPORTIONAL"`. Batch and grad_accum inherited from R1 config:
   ```python
   args = SentenceTransformerTrainingArguments(
       output_dir=cfg["multitask"]["output_dir"],
       num_train_epochs=cfg["multitask"]["epochs"],
       learning_rate=cfg["multitask"]["lr"],
       per_device_train_batch_size=cfg["contrastive_r1"]["per_device_batch_size"],  # 32
       gradient_accumulation_steps=cfg["contrastive_r1"]["grad_accum_steps"],       # 2
       lr_scheduler_type="cosine",
       warmup_ratio=0.1,
       bf16=cfg["hardware"]["bf16"],
       multi_dataset_batch_sampler="PROPORTIONAL",
       # ... other args
   )
   ```

8. **Train and save** to `models/vietlegal-e5-final/`.

- [ ] **Step 2: Launch training**

```bash
cd /home/mainq/vietlegal_e5 && accelerate launch --num_processes=2 scripts/train_multitask.py 2>&1 | head -30
```

- [ ] **Step 3: Validate**

```bash
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('models/vietlegal-e5-final')
print(f'Final model loaded. Dim: {m.get_sentence_embedding_dimension()}')
# Test Matryoshka truncation
import torch
emb = m.encode(['query: test'], convert_to_tensor=True)
for d in [1024, 512, 256, 128]:
    truncated = emb[:, :d]
    print(f'  Dim {d}: norm={torch.nn.functional.normalize(truncated, dim=1).shape}')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_multitask.py
git commit -m "feat: Stage 5 — multi-task blending (retrieval + classification + STS)"
```

---

### Task 11: Stage 6 — Evaluation (`scripts/evaluate.py`)

**Files:**
- Create: `scripts/evaluate.py`

- [ ] **Step 1: Write `scripts/evaluate.py`**

1. Load config, init W&B run `"evaluation"`.

2. **Load Zalo Legal dataset** and transform to IR evaluator format. The dataset contains: `question` (query text), `article_id` (relevant article IDs), and corpus articles with `article_id` + `text`:
   ```python
   ds = load_dataset(cfg["eval"]["zalo_dataset"])
   from scripts.utils import add_query_prefix, add_passage_prefix

   # Build corpus: article_id -> prefixed article text
   corpus = {}
   for row in ds["corpus"]:  # or appropriate split name
       corpus[str(row["article_id"])] = add_passage_prefix(row["text"])

   # Build queries and relevant_docs
   queries = {}
   relevant_docs = {}
   for i, row in enumerate(ds["queries"]):  # or appropriate split name
       qid = str(i)
       queries[qid] = add_query_prefix(row["question"])
       # article_id may be a list of relevant article IDs
       rel_ids = row["article_id"] if isinstance(row["article_id"], list) else [row["article_id"]]
       relevant_docs[qid] = {str(aid): 1 for aid in rel_ids}

   logger.info(f"Eval corpus: {len(corpus)} articles, {len(queries)} queries")
   ```

   **Note**: The exact split names and column names may vary — inspect the dataset with `print(ds)` and adapt. The schema above is based on the spec (lines 340-343). Also attempt to load `hiieu/legal_eval_label` as a fallback if the primary dataset is unavailable. ALQAC retrieval subtask is deferred to a future iteration.

3. **Define models to evaluate**:
   ```python
   models_to_eval = {}
   # Our checkpoints
   if cfg["eval"]["eval_all_checkpoints"]:
       for name, path in [
           ("vietlegal-e5-tsdae", cfg["tsdae"]["output_dir"]),
           ("vietlegal-e5-r1", cfg["contrastive_r1"]["output_dir"]),
           ("vietlegal-e5-r2", cfg["contrastive_r2"]["output_dir"]),
           ("vietlegal-e5-final", cfg["multitask"]["output_dir"]),
       ]:
           if Path(path).exists():
               models_to_eval[name] = path
   # Baselines from config
   for baseline in cfg["eval"]["baselines"]:
       models_to_eval[baseline.split("/")[-1]] = baseline
   ```

4. **Evaluate each model at each Matryoshka dim**:
   ```python
   from sentence_transformers.evaluation import InformationRetrievalEvaluator

   all_results = {}
   for model_name, model_path in models_to_eval.items():
       model = SentenceTransformer(model_path)
       dims = cfg["eval"]["matryoshka_dims"]
       # For baseline models, only eval at their native dim
       if "halong" in model_name:
           dims = cfg["eval"]["halong_eval_dims"]
       elif model_name not in ["vietlegal-e5-tsdae", "vietlegal-e5-r1", "vietlegal-e5-r2", "vietlegal-e5-final"]:
           dims = [model.get_sentence_embedding_dimension()]

       for dim in dims:
           evaluator = InformationRetrievalEvaluator(
               queries=queries, corpus=corpus, relevant_docs=relevant_docs,
               name=f"{model_name}_dim{dim}",
               truncate_dim=dim,
               ndcg_at_k=[10], mrr_at_k=[10], accuracy_at_k=[1, 3, 5, 10],
               recall_at_k=[10], map_at_k=[100],
               show_progress_bar=True,
           )
           results = evaluator(model)
           all_results[f"{model_name}_dim{dim}"] = results
   ```

5. **BM25 baseline** (if `cfg["eval"]["include_bm25"]`):
   ```python
   from rank_bm25 import BM25Okapi
   # Tokenize corpus, build BM25 index
   # For each query, retrieve top-10, compute NDCG@10
   # Add to all_results
   ```

6. **Save and report**:
   ```python
   import json
   with open(Path(cfg["paths"]["eval_dir"]) / "results.json", "w") as f:
       json.dump(all_results, f, indent=2)
   # Print formatted table
   # Log to W&B
   ```

- [ ] **Step 2: Run evaluation**

```bash
cd /home/mainq/vietlegal_e5 && python scripts/evaluate.py
```

- [ ] **Step 3: Inspect results**

```bash
python -c "
import json
results = json.load(open('eval/results.json'))
for model, metrics in sorted(results.items()):
    ndcg = metrics.get('ndcg@10', metrics.get('cosine_ndcg@10', 'N/A'))
    print(f'{model:40s} NDCG@10: {ndcg}')
"
```

Target: `vietlegal-e5-final_dim1024` NDCG@10 > 0.8976.

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: Stage 6 — evaluation on Zalo Legal with all checkpoints and baselines"
```

---

### Task 12: Stage 7 — Export & Deploy (`scripts/export.py`)

**Files:**
- Create: `scripts/export.py`

- [ ] **Step 1: Write `scripts/export.py`**

1. Load config, determine best model path (default: `models/vietlegal-e5-final/`).

2. **Push to HuggingFace Hub**:
   ```python
   from huggingface_hub import HfApi
   model = SentenceTransformer(best_model_path)

   # Create model card content
   model_card = f"""---
   language: vi
   tags:
   - sentence-transformers
   - embedding
   - legal
   - vietnamese
   - matryoshka
   license: apache-2.0
   ---

   # VietLegal-E5

   Vietnamese legal domain embedding model fine-tuned from `intfloat/multilingual-e5-large`.

   ## Usage

   **IMPORTANT**: This model requires prefixes:
   - Queries: `"query: "` + your query
   - Passages/Documents: `"passage: "` + your text

   ```python
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("{cfg['export']['hub_model_name']}")
   queries = model.encode(["query: quyền sở hữu trí tuệ là gì?"])
   passages = model.encode(["passage: Điều 1. Luật Sở hữu trí tuệ..."])
   ```

   ## Matryoshka Dimensions
   Supports truncation to: 1024, 512, 256, 128 dimensions.

   ## Benchmarks
   See eval/results.json for full comparison.
   """

   model.push_to_hub(cfg["export"]["hub_model_name"], private=False)
   ```

3. **ONNX export with FP16 quantization** (if `cfg["export"]["onnx_export"]`):
   ```python
   from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
   from optimum.onnxruntime.configuration import AutoQuantizationConfig
   from transformers import AutoTokenizer

   onnx_path = Path(cfg["paths"]["model_dir"]) / "vietlegal-e5-onnx"

   # Step 1: Export to ONNX
   ort_model = ORTModelForFeatureExtraction.from_pretrained(
       best_model_path, export=True
   )
   tokenizer = AutoTokenizer.from_pretrained(best_model_path)
   ort_model.save_pretrained(onnx_path)
   tokenizer.save_pretrained(onnx_path)

   # Step 2: FP16 quantization (if configured)
   if cfg["export"]["onnx_quantize_fp16"]:
       from onnxruntime.transformers import optimizer
       from onnxruntime.transformers.float16 import convert_float_to_float16
       import onnx
       onnx_model_path = onnx_path / "model.onnx"
       model_fp16 = onnx.load(str(onnx_model_path))
       model_fp16 = convert_float_to_float16(model_fp16)
       onnx.save(model_fp16, str(onnx_path / "model_fp16.onnx"))
       logger.info(f"FP16 ONNX model saved to {onnx_path / 'model_fp16.onnx'}")
   ```

4. **Validate ONNX**:
   ```python
   from optimum.onnxruntime import ORTModelForFeatureExtraction
   ort_model = ORTModelForFeatureExtraction.from_pretrained(onnx_path)
   tokenizer = AutoTokenizer.from_pretrained(onnx_path)
   inputs = tokenizer("passage: Điều 1. Phạm vi điều chỉnh", return_tensors="pt")
   outputs = ort_model(**inputs)
   print(f"ONNX output shape: {outputs.last_hidden_state.shape}")
   ```

5. **TEI compatibility check** (optional manual step):
   ```bash
   # Test with text-embeddings-inference Docker container
   # docker run --gpus all -p 8080:80 -v $(pwd)/models/vietlegal-e5-onnx:/model \
   #   ghcr.io/huggingface/text-embeddings-inference:latest --model-id /model
   logger.info("TEI compatibility: verify manually with TEI Docker container. See command above.")
   ```

- [ ] **Step 2: Run export**

```bash
cd /home/mainq/vietlegal_e5 && python scripts/export.py
```

- [ ] **Step 3: Verify HF Hub upload**

```bash
python -c "
from sentence_transformers import SentenceTransformer
# Test loading from Hub
# model = SentenceTransformer('your-username/vietlegal-e5')
# print('Hub model loaded successfully')
print('Verify manually at https://huggingface.co/your-username/vietlegal-e5')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/export.py
git commit -m "feat: Stage 7 — HF Hub push and ONNX export"
```

---

## Execution Order

Tasks must be executed sequentially — each depends on outputs from the previous:

```
Task 1 (scaffold) → Task 2 (utils) → Task 3 (Stage 0: data)
    → Task 4 (Stage 1: TSDAE) → Task 5 (Stage 2.1: prep training)
    → Task 6 (Stage 2.4: synth gen) → Task 7 (Stage 3: R1)
    → Task 8 (Stage 4.1-4.2: mine) → Task 9 (Stage 4.3: R2)
    → Task 10 (Stage 5: multitask) → Task 11 (Stage 6: eval)
    → Task 12 (Stage 7: export)
```

**Note on long-running tasks**: Tasks 3, 4, 6, 7, 8, 9, 10, 11 involve either large data processing or GPU training that takes hours. The "verify" steps in these tasks should be checked after the process completes. The script-writing and committing can happen immediately; validation is deferred.

**GPU resource constraint**: Tasks 4, 7, 9, 10 (training) and Task 6 (vLLM) each require the full 2x H100 GPUs. They cannot overlap. Task 6 must shut down vLLM before Task 7 starts.
