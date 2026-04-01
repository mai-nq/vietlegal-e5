# VietLegal-E5 — Design Specification

## Overview

Fine-tune `intfloat/multilingual-e5-large` (560M params, 1024-dim, XLM-R-large backbone) for Vietnamese legal document retrieval. The goal is to surpass `contextboxai/halong_embedding` (NDCG@10 = 0.8976 on Zalo Legal 20% sample).

## Decisions

| Decision | Choice |
|----------|--------|
| Scope | All 7 stages |
| GPU | 2x H100 80GB |
| Synthetic data | Yes — Qwen3 via vLLM, script-managed |
| Orchestration | Standalone scripts per stage |
| Config | Single YAML config file |
| Experiment tracking | Weights & Biases |
| Cross-encoder for hard neg filtering | Yes — BAAI/bge-reranker-v2-m3 |
| Pipeline architecture | Monolithic scripts + shared utils.py |
| Evaluation | All checkpoints (TSDAE, R1, R2, final) + baselines |

---

## Project Structure

```
vietlegal-e5/
├── config.yaml                    # All hyperparams, paths, flags
├── requirements.txt               # Pinned dependencies
├── scripts/
│   ├── utils.py                   # Config loader, prefix helpers, W&B init, logging
│   ├── prepare_data.py            # Stage 0: chunk legal docs → chunks.parquet
│   ├── train_tsdae.py             # Stage 1: TSDAE domain adaptation
│   ├── prepare_training.py        # Stage 2: load query-passage pairs, split, prefix
│   ├── generate_queries.py        # Stage 2.4: vLLM + Qwen3 synthetic query gen
│   ├── train_contrastive.py       # Stage 3: contrastive fine-tuning R1
│   ├── mine_hard_negatives.py     # Stage 4.1-4.2: encode, FAISS index, mine + cross-encoder filter
│   ├── train_hard_neg.py          # Stage 4.3: contrastive R2 with hard negatives
│   ├── train_multitask.py         # Stage 5: multi-task blending
│   ├── evaluate.py                # Stage 6: eval on Zalo Legal + baselines
│   └── export.py                  # Stage 7: push to HF Hub, ONNX export
├── data/                          # Generated data artifacts
│   ├── chunks.parquet             # Stage 0 output
│   ├── synthetic_queries.parquet  # Stage 2.4 output
│   ├── train.parquet              # Stage 2 output (merged with synthetic)
│   ├── val.parquet
│   ├── test.parquet
│   └── hard_negatives.parquet     # Stage 4 output
├── models/                        # Saved model checkpoints
│   ├── vietlegal-e5-tsdae/       # Stage 1
│   ├── vietlegal-e5-r1/          # Stage 3
│   ├── vietlegal-e5-r2/          # Stage 4
│   ├── vietlegal-e5-final/       # Stage 5
│   └── vietlegal-e5-onnx/        # Stage 7
└── eval/                          # Evaluation results
    └── results.json
```

## Config Schema (`config.yaml`)

```yaml
paths:
  base_dir: "."
  data_dir: "data"
  model_dir: "models"
  eval_dir: "eval"

data:
  legal_docs_dataset: "th1nhng0/vietnamese-legal-documents"
  query_pairs_dataset: "phamson02/large-vi-legal-queries"
  min_content_chars: 100
  chunk_max_tokens: 512
  chunk_window_tokens: 256
  chunk_overlap_tokens: 64
  chunk_min_tokens: 50

tsdae:
  base_model: "intfloat/multilingual-e5-large"
  lr: 3e-5
  batch_size: 32
  epochs: 1
  max_seq_len: 512
  warmup_steps: 1000
  weight_decay: 0.01
  scheduler: "warmup_linear"
  output_dir: "models/vietlegal-e5-tsdae"

contrastive_r1:
  lr: 1e-5
  per_device_batch_size: 32  # plan.md says 64/device but that gives effective=256, not 128. Using 32 to match target effective=128.
  grad_accum_steps: 2
  epochs: 3
  warmup_ratio: 0.1
  scheduler: "cosine"
  matryoshka_dims: [1024, 512, 256, 128]
  eval_steps: 1000
  save_total_limit: 3
  output_dir: "models/vietlegal-e5-r1"

hard_neg:
  top_k: 100
  neg_range_start: 50
  neg_range_end: 100
  use_cross_encoder: true
  cross_encoder_model: "BAAI/bge-reranker-v2-m3"
  cross_encoder_threshold: 0.5
  max_negatives_per_query: 3
  faiss_index_type: "flat"  # "flat" or "ivf"
  faiss_ivf_nlist: 4096
  encode_batch_size: 512

contrastive_r2:
  lr: 5e-6
  epochs: 2
  output_dir: "models/vietlegal-e5-r2"
  # inherits per_device_batch_size, grad_accum_steps, matryoshka_dims from r1

multitask:
  sampling_ratios:
    retrieval: 0.7
    classification: 0.2
    sts: 0.1
  lr: 5e-6
  epochs: 1
  output_dir: "models/vietlegal-e5-final"

synth:
  model_name: "Qwen/Qwen3-Next-80B-A3B-Instruct"
  fallback_model: "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"  # if bf16 doesn't fit in VRAM
  vllm_tensor_parallel: 2
  vllm_max_model_len: 4096
  vllm_port: 8000
  num_queries_per_chunk: 3
  batch_size: 64
  checkpoint_every_n: 10000
  max_new_tokens: 512
  temperature: 0.7

eval:
  zalo_dataset: "GreenNode/zalo-ai-legal-text-retrieval-vn"
  matryoshka_dims: [1024, 512, 256, 128]
  eval_all_checkpoints: true
  baselines:
    - "intfloat/multilingual-e5-large"
    - "intfloat/multilingual-e5-base"
    - "contextboxai/halong_embedding"
    - "bkai-foundation-models/vietnamese-bi-encoder"
    - "hmthanh/VietnamLegalText-SBERT"
  include_bm25: true
  halong_eval_dims: [768, 256]  # eval halong_embedding at full and Matryoshka-truncated dims

export:
  hub_model_name: "vietlegal-e5"
  onnx_export: true
  onnx_quantize_fp16: true

wandb:
  project: "vietlegal-e5"
  tags: ["legal", "vietnamese", "embedding", "me5-large"]

hardware:
  num_gpus: 2
  bf16: true
  dataloader_workers: 4
  seed: 42

split:
  train_ratio: 0.95
  val_ratio: 0.025
  test_ratio: 0.025
  seed: 42
```

---

## Stage 0 — Data Preparation (`prepare_data.py`)

**Input**: `th1nhng0/vietnamese-legal-documents` (518K docs, 3.6GB)

**Pipeline**:
1. Load both `metadata` and `content` configs via `datasets.load_dataset()`, join on `id` column
2. Filter: remove docs with `len(content) < 100` chars, deduplicate by `document_number`
3. Chunk (priority order):
   - Primary: regex split on `Điều \d+` — each article becomes one chunk
   - Fallback: if chunk > 512 tokens or no regex match → sliding window (256 tokens, 64 overlap)
   - Merge: chunks < 50 tokens get merged with adjacent chunk
   - Tokenizer: `xlm-roberta-large` tokenizer (same as mE5-large) for token counting
4. Attach metadata: `legal_type`, `legal_sectors`, `source_doc_id`, `chunk_index`
5. Output: `data/chunks.parquet`

**Output columns**: `chunk_id`, `text`, `source_doc_id`, `legal_type`, `legal_sectors`, `chunk_index`

**Expected size**: ~2-5M chunks

---

## Stage 1 — TSDAE Pre-training (`train_tsdae.py`)

**Purpose**: Domain-adapt mE5-large to Vietnamese legal vocabulary before contrastive training.

**Input**: All chunk texts from `data/chunks.parquet`

**Method**: TSDAE (Transformer-based Sequential Denoising Auto-Encoder)
- Corrupts input by deleting ~60% of tokens randomly
- Model must reconstruct the original sentence from the corrupted version
- Unsupervised — no labels needed

**Implementation**:
1. Load `intfloat/multilingual-e5-large` as SentenceTransformer
2. Create a standard HuggingFace `Dataset` from chunk texts (single column `"text"`)
3. Use `DenoisingAutoEncoderLoss` with `tie_encoder_decoder=True` (uses the XLM-R backbone as both encoder and decoder)
4. Provide dataset with a single `sentence_0` column containing clean text. `DenoisingAutoEncoderLoss` handles corruption internally (deletes ~60% of tokens). Do NOT apply manual noise via `set_transform()` — that would cause double-corruption.
5. Train with `SentenceTransformerTrainer`:
   - lr=3e-5, per_device_batch=32 (effective=64 with 2 GPUs), epochs=1, max_seq_len=512
   - warmup=1000 steps, weight_decay=0.01, scheduler=warmup+linear
   - bf16=True, W&B logging
6. Multi-GPU: `accelerate launch --num_processes=2` for 2x H100 DDP
7. Save to `models/vietlegal-e5-tsdae/`

**Estimated time**: ~3M chunks, per_device_batch=32 (effective=64), 1 epoch ≈ ~47K steps. On 2x H100, roughly 6-8 hours.

---

## Stage 2 — Training Data Preparation

### 2.1 — `prepare_training.py`

**Input**: `phamson02/large-vi-legal-queries` (~507K pairs)

1. Load dataset from HuggingFace
2. Add prefixes: `"query: "` + query, `"passage: "` + positive passage
3. Split: 95% train, 2.5% val, 2.5% test, seed=42
4. Save to `data/train.parquet`, `data/val.parquet`, `data/test.parquet`

### 2.4 — `generate_queries.py` (Synthetic Generation)

**Input**: `data/chunks.parquet` from Stage 0

1. Launch vLLM server with `Qwen/Qwen3-Next-80B-A3B-Instruct` as subprocess
2. Wait for server health check (poll `/health` endpoint)
3. For each chunk, send prompt via OpenAI-compatible API:
   ```
   Bạn là chuyên gia pháp luật Việt Nam. Dựa trên đoạn văn bản pháp luật sau,
   hãy viết đúng 3 câu hỏi pháp lý mà đoạn văn này có thể trả lời.
   Câu hỏi phải tự nhiên, đa dạng về cách hỏi.

   Đoạn văn: {chunk_text}
   ```
4. Parse responses → create (query, passage) pairs with prefixes
5. Batch processing with async requests for throughput
6. Checkpoint progress every 10K chunks for resumability
7. Save to `data/synthetic_queries.parquet`
8. Merge synthetic data into training set only:
   - Load existing `data/train.parquet` (from `prepare_training.py`)
   - Concatenate synthetic queries with train split only
   - **Do NOT modify val/test splits** — they remain real data only to avoid contamination
   - Shuffle and save updated `data/train.parquet`
   - Val/test remain unchanged from `prepare_training.py` output
9. Shutdown vLLM server
10. Final output: updated `data/train.parquet` (~1M+ pairs), unchanged `data/val.parquet`, `data/test.parquet`

---

## Stage 3 — Contrastive Fine-tuning R1 (`train_contrastive.py`)

**Input**: Train/val splits from Stage 2
**Base model**: `models/vietlegal-e5-tsdae/` (or raw mE5-large if TSDAE skipped — config flag)

**Training**:
1. Load base model as `SentenceTransformer`
2. Build dataset from (query, positive) pairs — already prefixed
3. Loss: `MatryoshkaLoss(MultipleNegativesRankingLoss)`, dims=[1024, 512, 256, 128]
4. `SentenceTransformerTrainer`:
   - lr=1e-5, per_device_batch=32, grad_accum=2 → effective batch=128 (32 × 2 GPUs × 2 accum)
   - epochs=3, warmup_ratio=0.1, scheduler=cosine, bf16=True
   - eval_steps=1000, save_total_limit=3, load_best_model_at_end=True
5. Evaluator: `InformationRetrievalEvaluator` on val set, NDCG@10
6. Multi-GPU: `accelerate launch --num_processes=2`
7. W&B logging
8. Save best model to `models/vietlegal-e5-r1/`

**Estimated time**: ~481K pairs × 3 epochs, batch 128 ≈ ~11.3K steps/epoch ≈ ~34K total. On 2x H100, roughly 6-10 hours.

---

## Stage 4 — Hard Negative Mining + Round 2

### 4.1-4.2 — `mine_hard_negatives.py`

**Input**: R1 model, all passages from training data

1. Load R1 model
2. Encode all passages (with `"passage: "` prefix), batch=512, bf16
3. Build FAISS index: `IndexFlatIP` for cosine similarity (vectors normalized). Use `IndexIVFFlat` (nlist=4096) if corpus > 5M passages.
4. For each training query: retrieve top-100 → remove true positives → take rank 50-100 as candidates
5. Cross-encoder reranking with `BAAI/bge-reranker-v2-m3`:
   - Score each (query, candidate) pair
   - Filter candidates with score > 0.5 (likely false negatives)
   - Take top 1-3 remaining as hard negatives
6. Output: `data/hard_negatives.parquet` — columns: `query`, `positive`, `hard_negative`

### 4.3 — `train_hard_neg.py`

**Base model**: `models/vietlegal-e5-r1/`

- Loss: `MatryoshkaLoss(MultipleNegativesRankingLoss)` with triplets (query, pos, neg)
- lr=5e-6, epochs=2, other params from R1 config
- Save to `models/vietlegal-e5-r2/`

---

## Stage 5 — Multi-task Blending (`train_multitask.py`)

**Input**: R2 model from `models/vietlegal-e5-r2/`

**Datasets**:
1. **Retrieval (70%)**: Query-passage pairs from Stage 2 → `MultipleNegativesRankingLoss`
2. **Classification (20%)**: Chunks grouped by `legal_type` from Stage 0 metadata. Same-type chunks are positives → `MultipleNegativesRankingLoss`
3. **STS (10%)**: Use `stsb_multi_mt` Vietnamese split from HuggingFace → `CoSENTLoss`. If unavailable or too small, fallback: skip STS, redistribute to 80/20 retrieval/classification.

**Implementation** (ST v3 multi-task API):
1. Create a `dict` of HuggingFace `Dataset` objects, one per task: `{"retrieval": ds_retrieval, "classification": ds_classification, "sts": ds_sts}`
2. Create a `dict` of losses: `{"retrieval": MatryoshkaLoss(MNRL), "classification": MatryoshkaLoss(MNRL), "sts": MatryoshkaLoss(CoSENTLoss)}`
3. To achieve 70/20/10 sampling ratio: oversample/undersample each dataset proportionally before passing to the trainer. E.g., if retrieval has 500K rows and we want 70%, classification 100K rows for 20%, set classification to ~143K rows via oversampling. Alternative: use `multi_dataset_batch_sampler="PROPORTIONAL"` and adjust dataset sizes accordingly.
4. Pass the dataset dict and loss dict to `SentenceTransformerTrainer` — it handles task-homogeneous batching automatically
5. lr=5e-6, epochs=1
6. Save to `models/vietlegal-e5-final/`

Classification dataset is constructed from `chunks.parquet` metadata — group chunks by `legal_type`, sample pairs where same-type chunks are positives. No external labeling needed.

**Note on clustering**: The original plan mentioned a clustering task (chunks grouped by `legal_sectors`). This is intentionally excluded to keep multi-task simpler — `legal_sectors` overlap significantly with `legal_type`, and adding a third contrastive task risks diluting retrieval performance. Can be revisited if classification performance is insufficient.

---

## Stage 6 — Evaluation (`evaluate.py`)

**Benchmarks**:
1. Primary: Zalo AI 2021 Legal Text Retrieval — `GreenNode/zalo-ai-legal-text-retrieval-vn` (~114K articles, 818 queries)
2. Secondary: ALQAC retrieval subtask (if available)

**Zalo dataset schema**: The dataset contains `question` (query text), `article_id` (relevant article IDs), and corpus articles with `article_id` + `text`. Transform into:
- `queries`: `{qid: "query: " + question_text}`
- `corpus`: `{article_id: "passage: " + article_text}`
- `relevant_docs`: `{qid: {article_id: 1 for each relevant article}}`

**Protocol**:
1. Load eval dataset, transform into queries/corpus/relevant_docs dicts as above for `InformationRetrievalEvaluator`
2. Prefixes are applied during dict construction (see above)
3. Evaluate at all Matryoshka dims: 1024, 512, 256, 128
4. Metrics: NDCG@10 (primary), Accuracy@1/@3/@5/@10, Recall@10, MRR@10, MAP@100

**Models evaluated**:
- Our checkpoints: TSDAE, R1, R2, final (at all dims)
- Baselines: mE5-large (zero-shot), mE5-base (zero-shot), halong_embedding (768d + 256d), vietnamese-bi-encoder, VietnamLegalText-SBERT, BM25

**Output**: `eval/results.json` + console table + W&B metrics

**Target**: Beat halong_embedding NDCG@10 = 0.8976

---

## Stage 7 — Export & Deploy (`export.py`)

1. Push best model to HuggingFace Hub as `{username}/vietlegal-e5`
   - Include model card with benchmarks, training details, usage examples
   - Prominently document prefix requirement (`"query: "` / `"passage: "`)
2. ONNX export via `optimum` library, FP16 quantization
3. TEI compatibility validation

---

## Dependencies

```
sentence-transformers>=3.0
transformers>=4.40
datasets
torch>=2.1
accelerate
faiss-gpu
wandb
pandas
pyarrow
pyyaml
rank_bm25
optimum[onnxruntime]
huggingface_hub
openai  # for vLLM OpenAI-compatible client
vllm    # for synthetic query generation
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| GPU VRAM overflow with mE5-large | Reduce batch + increase grad_accum. Fallback to mE5-base |
| Catastrophic forgetting general tasks | Blend NLI/STS data in Stage 5 |
| Forgotten prefix `"query:"` / `"passage:"` | Hard-code in utils.py, validate before training |
| Low-quality synthetic queries | Spot-check 100 random pairs before training |
| Hard negatives contain false negatives | Cross-encoder filtering + rank 50-100 only |
| Overfitting on small eval set | Eval multiple datasets, monitor train/val loss |
| vLLM server crash during synthetic gen | Checkpoint every 10K chunks, auto-resume |
| Qwen3-80B may not fit in 2x H100 for vLLM | Use FP8 quantized variant (`Qwen3-Next-80B-A3B-Instruct-FP8`), or reduce to smaller model, or use external API. Note: 80B params in bf16 = ~160GB, leaves minimal KV cache headroom on 2x80GB |
| Training diverges (NaN loss) or OOM mid-run | Use `resume_from_checkpoint=True` in trainer args. Monitor loss via W&B alerts. If NaN: halve LR and restart from last checkpoint. If OOM: reduce per_device_batch and increase grad_accum |
| GPU resource conflict between vLLM and training | Stage 2.4 (vLLM) must complete and server must shut down before any training stage. Stages are sequential by design — never overlap |

---

## Hyperparams Summary

| Stage | LR | Effective Batch | Epochs | Loss |
|-------|-----|-----------------|--------|------|
| 1 TSDAE | 3e-5 | 64 (32/device × 2 GPUs) | 1 | DenoisingAutoEncoderLoss |
| 3 Contrastive R1 | 1e-5 | 128 (32/device × 2 GPUs × 2 accum) | 3 | Matryoshka(MNRL) |
| 4 Hard Neg R2 | 5e-6 | 128 (32/device × 2 GPUs × 2 accum) | 2 | Matryoshka(MNRL) |
| 5 Multi-task | 5e-6 | 128 (32/device × 2 GPUs × 2 accum) | 1 | Mixed (MNRL + CoSENT) |
