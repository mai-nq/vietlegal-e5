# VietLegal-E5 — Experiment Guide

## Goal

Fine-tune `intfloat/multilingual-e5-large` (560M params, 1024-dim) cho Vietnamese legal domain.
Target: vượt baseline `contextboxai/halong_embedding` (NDCG@10 = 0.8976 trên Zalo Legal 20% sample).

---

## Architecture Decision

- **Base model**: `intfloat/multilingual-e5-large` (Option A — backbone lớn, 1024-dim, XLM-R-large)
- **Không dùng** `contextboxai/halong_embedding` làm base vì nó chỉ là mE5-base (278M, 768-dim), headroom thấp hơn
- **Loss**: MatryoshkaLoss wrapping MultipleNegativesRankingLoss → output dims [1024, 512, 256, 128]
- **Framework**: Sentence Transformers v3 (`SentenceTransformerTrainer`)
- **Prefix convention**: `"query: "` cho queries, `"passage: "` cho passages (bắt buộc với mE5)

---

## Stage 0 — Data Preparation

**Input**: `th1nhng0/vietnamese-legal-documents` (518K docs, 3.6GB, CC BY 4.0)

**Bước thực hiện**:

1. Load cả 2 configs (`metadata` + `content`), join bằng cột `id`
2. Filter: loại docs có content < 100 ký tự, loại duplicates theo `document_number`
3. Chunking theo thứ tự ưu tiên:
   - Regex split theo `Điều \d+` (mỗi điều = 1 chunk)
   - Nếu chunk > 512 tokens hoặc không match regex → sliding window 256 tokens, overlap 64 tokens
   - Chunk < 50 tokens → merge với chunk liền kề
4. Gắn metadata vào mỗi chunk: `legal_type`, `legal_sectors`, `source_doc_id`
5. Output: `chunks.parquet` — target ~2–5M chunks

---

## Stage 1 — Domain-Adaptive Pre-training (TSDAE)

**Mục tiêu**: Adapt model vào Vietnamese legal vocabulary trước contrastive training.

**Input**: Toàn bộ chunks từ Stage 0 (chỉ cần text, không cần labels)

**Method**: TSDAE (Transformer-based Sequential Denoising Auto-Encoder) — unsupervised

**Base model**: `intfloat/multilingual-e5-large`

**Params**: lr=3e-5, batch=32, epochs=1, max_seq_len=512, warmup=1000 steps, weight_decay=0.01, scheduler=warmup+linear

**Nếu không đủ GPU/thời gian**: có thể skip stage này, đi thẳng Stage 2. TSDAE cho thêm ~1–2 NDCG points nhưng không critical.

**Output**: `models/vietlegal-e5-tsdae/`

---

## Stage 2 — Load & Prepare Training Data

### 2.1 Primary data

- Dataset: `phamson02/large-vi-legal-queries` (~507K synthetic query-passage pairs, sinh bởi Llama3-70B)
- Format: mỗi row có `query` và `positive` (passage)

### 2.2 Split

- Train 95% (~481K), Val 2.5% (~12.5K), Test 2.5% (~12.5K), seed=42

### 2.3 Prefix — CRITICAL

- Thêm `"query: "` vào đầu mỗi query
- Thêm `"passage: "` vào đầu mỗi passage
- **Không được quên prefix** — mE5 train với prefix, bỏ sẽ drop performance nghiêm trọng

### 2.4 Optional: sinh thêm queries

- Dùng LLM Qwen/Qwen3-Next-80B-A3B-Instruct, serving bằng vllm ở trên servers luôn sinh 3–5 câu hỏi pháp lý cho mỗi chunk từ Stage 0
- Prompt: yêu cầu model viết câu hỏi mà chunk đó có thể trả lời
- Lấy thêm khoảng: 500k samples nữa
---

## Stage 3 — Contrastive Fine-tuning (Round 1)

**Input**: Train/val splits từ Stage 2

**Base model**: output Stage 1 (hoặc raw mE5-large nếu skip Stage 1)

**Loss**: MatryoshkaLoss(MultipleNegativesRankingLoss), dims=[1024, 512, 256, 128]

**Params**:
- lr=1e-5, batch=64/device, grad_accum=2 (effective batch=128)
- epochs=3, warmup_ratio=0.1, scheduler=cosine, bf16=True
- eval every 1000 steps, save top 3 checkpoints

**GPU**: 2x H100 80GB

**Output**: `models/vietlegal-e5-r1/`

---

## Stage 4 — Hard Negative Mining + Round 2

### 4.1 Encode corpus

- Dùng model Stage 3 encode toàn bộ passages (với prefix `"passage: "`)
- Build FAISS index (IndexFlatIP hoặc IndexIVFFlat)

### 4.2 Mine hard negatives

- Cho mỗi query → retrieve top-100 passages
- Loại true positives
- Lấy rank 50–100 làm hard negatives (tránh top ranks — nhiều false negatives)
- Optional: cross-encoder rerank để filter false negatives

### 4.3 Train Round 2

- Data: triplets (query, positive, hard_negative)
- Loss: giữ nguyên MatryoshkaLoss(MNRL)
- lr=**5e-6** (thấp hơn R1), epochs=1–2, params khác giữ nguyên

**Output**: `models/vietlegal-e5-r2/`

---

## Stage 5 — Multi-task Blending (Optional)

Mục tiêu: cải thiện classification/clustering mà không giảm retrieval.

### Datasets bổ sung

- **Classification**: pairs `(chunk_text, legal_type)` từ metadata Stage 0 — chunks cùng legal_type là positives
- **STS**: Vietnamese STS / paraphrase pairs nếu có → CoSENTLoss
- **Clustering**: chunks cùng `legal_sectors` là positives

### Training

- Task-homogeneous batching: mỗi batch từ 1 dataset duy nhất
- Tỷ lệ sampling: 70% retrieval, 20% classification, 10% STS
- lr=5e-6, epochs=1

**Output**: `models/vietlegal-e5-final/`

---

## Stage 6 — Evaluation

### Eval datasets

1. **Zalo AI 2021 Legal Text Retrieval** — ~114K articles, 818 queries (benchmark chính)
   - Dataset: `GreenNode/zalo-ai-legal-text-retrieval-vn` hoặc `hiieu/legal_eval_label`
2. **ALQAC** retrieval subtask nếu available

### Metrics

- **Primary**: NDCG@10
- Secondary: Accuracy@1, @3, @5, @10, Recall@10, MRR@10, MAP@100

### Baselines

| Model | Params | Dim | Notes |
|-------|--------|-----|-------|
| `intfloat/multilingual-e5-large` | 560M | 1024 | Zero-shot |
| `intfloat/multilingual-e5-base` | 278M | 768 | Zero-shot |
| `contextboxai/halong_embedding` | 278M | 768 | mE5-base, general VN, ~100K pairs |
| `contextboxai/halong_embedding` (256d) | 278M | 256 | Matryoshka truncated |
| `bkai-foundation-models/vietnamese-bi-encoder` | — | — | MS MARCO VN + Zalo Legal |
| `hmthanh/VietnamLegalText-SBERT` | 135M | — | PhoBERT, VN legal |
| BM25 (pyserini) | — | — | Lexical baseline |

### Eval protocol

- Dùng `InformationRetrievalEvaluator` từ Sentence Transformers
- Eval ở nhiều Matryoshka dims: 1024, 512, 256, 128
- Report bảng so sánh tất cả baselines + tất cả dims

---

## Stage 7 — Export & Deploy

- Push best model lên HuggingFace Hub dưới tên `vietlegal-e5`
- Export ONNX nếu cần inference nhanh
- Test với `text-embeddings-inference` (TEI) cho production

---

## File Structure

```
vietlegal-e5/
├── EXPERIMENT_GUIDE.md
├── data/
│   ├── chunks.parquet              ← Stage 0
│   ├── train.parquet               ← Stage 2
│   ├── val.parquet
│   └── test.parquet
├── models/
│   ├── vietlegal-e5-tsdae/         ← Stage 1
│   ├── vietlegal-e5-r1/            ← Stage 3
│   ├── vietlegal-e5-r2/            ← Stage 4
│   └── vietlegal-e5-final/         ← Stage 5
├── eval/
│   └── results.json                ← Stage 6
└── scripts/
    ├── prepare_data.py
    ├── train_tsdae.py
    ├── train_contrastive.py
    ├── mine_hard_negatives.py
    ├── train_multitask.py
    └── evaluate.py
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| GPU VRAM không đủ cho mE5-large | Giảm batch + tăng grad_accum, hoặc fallback mE5-base → `vietlegal-e5-base` |
| Catastrophic forgetting general tasks | Blend NLI/STS data vào Stage 5 |
| Quên prefix `"query:"` / `"passage:"` | Hard-code trong pipeline, validate trước khi train |
| Synthetic queries kém chất lượng | Spot-check 100 random pairs trước khi train |
| Hard negatives chứa false negatives | Cross-encoder filter, chỉ lấy rank 50–100 |
| Overfitting trên small eval set | Eval nhiều datasets, monitor train/val loss |

---

## Hyperparams Summary

| Stage | LR | Effective Batch | Epochs | Loss |
|-------|-----|-----------------|--------|------|
| 1 TSDAE | 3e-5 | 32 | 1 | DenoisingAutoEncoderLoss |
| 3 Contrastive R1 | 1e-5 | 128 | 3 | Matryoshka(MNRL) |
| 4 Hard Neg R2 | 5e-6 | 128 | 1–2 | Matryoshka(MNRL) |
| 5 Multi-task | 5e-6 | 128 | 1 | Mixed (MNRL + CoSENT) |