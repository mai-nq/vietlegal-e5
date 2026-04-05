# VietLegal Embedding Models

Vietnamese legal domain embedding models fine-tuned for legal text retrieval. This repository contains training code and pipelines for two model families:

- **VietLegal-Harrier-0.6B** - State-of-the-art, fine-tuned from `microsoft/harrier-oss-v1-0.6b`
- **VietLegal-E5** - Fine-tuned from `intfloat/multilingual-e5-large` with Matryoshka support

## Results

Evaluated on [MTEB ZacLegalTextRetrieval](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) (61.4K corpus documents, 818 test queries).

| Model | Params | NDCG@10 | MRR@10 | Recall@10 |
|-------|--------|---------|--------|-----------|
| **[vietlegal-harrier-0.6b](https://huggingface.co/mainguyen9/vietlegal-harrier-0.6b)** | 600M | **0.7813** | **0.7303** | **0.9321** |
| vietlegal-harrier-270m | 270M | 0.7174 | 0.6636 | 0.8864 |
| [vietlegal-e5](https://huggingface.co/mainguyen9/vietlegal-e5) (mE5-large) | 560M | 0.7310 | 0.6770 | 0.8972 |
| microsoft/harrier-oss-v1-0.6b | 600M | 0.7210 | - | - |
| intfloat/multilingual-e5-large | 560M | 0.6660 | - | - |
| bkai-foundation-models/vietnamese-bi-encoder | 135M | 0.6160 | - | - |
| contextboxai/halong_embedding | 278M | 0.6009 | - | - |

**Key highlights:**
- **VietLegal-Harrier-0.6B** achieves state-of-the-art on Vietnamese legal retrieval
- **+5.0 points NDCG@10** over vietlegal-e5 (mE5-large): 0.7813 vs 0.7310
- **+6.0 points NDCG@10** over original Harrier-0.6b: 0.7813 vs 0.7210
- VietLegal-E5 supports **Matryoshka embeddings** at [1024, 512, 256, 128] dims

## Usage

### VietLegal-Harrier-0.6B (Recommended)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("mainguyen9/vietlegal-harrier-0.6b")

# Harrier uses instruction-based queries
queries = ["Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: Thu tuc dang ky kinh doanh gom nhung buoc nao?"]
passages = ["Dieu 27. Trinh tu, thu tuc dang ky doanh nghiep..."]

q_emb = model.encode(queries)
p_emb = model.encode(passages)

similarity = q_emb @ p_emb.T
```

### VietLegal-E5 (with Matryoshka support)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("mainguyen9/vietlegal-e5")

# E5 models require "query: " and "passage: " prefixes
queries = ["query: Thu tuc dang ky kinh doanh gom nhung buoc nao?"]
passages = ["passage: Dieu 27. Trinh tu, thu tuc dang ky doanh nghiep..."]

q_emb = model.encode(queries)
p_emb = model.encode(passages)

# For Matryoshka - truncate to desired dimension
model.truncate_dim = 256
q_emb_256 = model.encode(queries)
```

## Training Pipeline

The model is trained through a multi-stage pipeline:

```
Stage 1: Data Preparation
│  518K Vietnamese legal docs → ~500K chunks (article-aware segmentation)
│  Dataset: th1nhng0/vietnamese-legal-documents
│  507K query-passage pairs from phamson02/large-vi-legal-queries
│
Stage 2: Contrastive Fine-tuning
│  MatryoshkaLoss(MultipleNegativesRankingLoss) at dims [1024,512,256,128]
│  → NDCG@10 = 0.6725
│
Stage 3: Hard Negative Mining
│  FAISS retrieval → mine rank 50-100 as hard negatives
│  Retrain with triplets (query, positive, hard_negative)
│
Stage 4: Multi-task Blending
│  70% retrieval + 20% legal-type classification + 10% STS
│  → NDCG@10 = 0.7229 (best)
│
Stage 5: Evaluation & Export
```

**Key insight**: Multi-task blending provides the largest gain (+5.04 NDCG points), combining retrieval with legal document classification and semantic similarity to prevent overfitting.

## Running the Pipeline

```bash
# Stage 1: Prepare data
python scripts/prepare_data.py
python scripts/prepare_training.py

# Stage 1.5 (optional): Generate synthetic queries
python scripts/generate_queries.py

# Stage 2: Contrastive fine-tuning
accelerate launch --num_processes=4 scripts/train_contrastive.py

# Stage 3: Mine hard negatives + retrain
python scripts/mine_hard_negatives.py
accelerate launch --num_processes=4 scripts/train_hard_neg.py

# Stage 4: Multi-task blending
accelerate launch --num_processes=4 scripts/train_multitask.py

# Stage 5: Evaluate & export
python scripts/evaluate.py
python scripts/export.py
```

## Data

| Dataset | Stage | Records | Description |
|---------|-------|---------|-------------|
| `chunks.parquet` | 1 | ~500K | Legal doc chunks with metadata |
| `train.parquet` | 1 | ~480K | Query-passage pairs (with prefixes) |
| `val.parquet` | 1 | ~12.5K | Validation split |
| `test.parquet` | 1 | ~12.5K | Test split |
| `hard_negatives.parquet` | 3 | ~1.5M | Mined triplets (193MB) |
| `synthetic_queries.parquet` | 1.5 | — | LLM-generated pairs (66MB, optional) |

## Configuration

All hyperparameters are centralized in `config.yaml`. Key settings:

- **Base model**: `intfloat/multilingual-e5-large`
- **Hardware**: 4x GPU (A100/H100 80GB), bf16 mixed precision
- **Matryoshka dims**: [1024, 512, 256, 128]
- **Tracking**: Weights & Biases (`vietlegal-e5` project)

## Requirements

- Python 3.11+
- PyTorch 2.1+
- sentence-transformers >= 3.0
- faiss-cpu >= 1.7
- mteb (for evaluation)
- vllm >= 0.4 (optional, for synthetic query generation)

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── config.yaml              # Central configuration
├── scripts/
│   ├── prepare_data.py      # Stage 1: Chunking
│   ├── prepare_training.py  # Stage 1: Data splits
│   ├── generate_queries.py  # Stage 1.5: Synthetic queries (optional)
│   ├── train_contrastive.py # Stage 2: Contrastive training
│   ├── mine_hard_negatives.py # Stage 3: Hard neg mining
│   ├── train_hard_neg.py    # Stage 3: Retrain with hard negs
│   ├── train_multitask.py   # Stage 4: Multi-task blending
│   ├── evaluate.py          # Stage 5: MTEB eval
│   ├── export.py            # Stage 5: Export
│   └── utils.py             # Shared utilities
├── data/                    # Training data (parquet)
├── models/                  # Model checkpoints
├── eval/                    # Evaluation results
└── logs/                    # Training logs
```
