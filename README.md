# VietLegal-E5

A Vietnamese legal domain embedding model fine-tuned from `intfloat/multilingual-e5-large` (560M params, 1024-dim). Achieves **NDCG@10 = 0.7229** on the Zalo AI Legal Text Retrieval benchmark, outperforming `microsoft/harrier-oss-v1-0.6b` (0.7210) and the zero-shot mE5-large baseline (0.6660) by **+5.69 points**.

Supports **Matryoshka embeddings** at dimensions [1024, 512, 256, 128] — the 128-dim output (0.7073) still beats mE5-large at full 1024-dim, enabling **8x compression** with no quality loss.

## Results

| Model | Params | Dim | NDCG@10 |
|-------|--------|-----|---------|
| **vietlegal-e5-final-v2** | 560M | 1024 | **0.7229** |
| vietlegal-e5-final-v2 | 560M | 512 | 0.7208 |
| vietlegal-e5-final-v2 | 560M | 256 | 0.7058 |
| vietlegal-e5-final-v2 | 560M | 128 | 0.7073 |
| microsoft/harrier-oss-v1-0.6b | 600M | 1024 | 0.7210 |
| intfloat/multilingual-e5-large | 560M | 1024 | 0.6660 |
| vietlegal-e5-r1-v2 | 560M | 1024 | 0.6725 |
| bkai-foundation-models/vietnamese-bi-encoder | 135M | 768 | 0.6160 |
| intfloat/multilingual-e5-base | 278M | 768 | 0.6030 |
| contextboxai/halong_embedding | 278M | 768 | 0.6009 |

Evaluated on [MTEB ZacLegalTextRetrieval](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) (61.4K corpus documents, 818 test queries).

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/vietlegal-e5-final-v2")

# Important: mE5 models require "query: " and "passage: " prefixes
queries = ["query: Thủ tục đăng ký kinh doanh gồm những bước nào?"]
passages = ["passage: Điều 27. Trình tự, thủ tục đăng ký doanh nghiệp..."]

q_emb = model.encode(queries)
p_emb = model.encode(passages)

# For Matryoshka — truncate to desired dimension
model.truncate_dim = 256
q_emb_256 = model.encode(queries)
```

## Training Pipeline

The model is trained through a multi-stage pipeline:

```
Stage 0: Data Preparation
│  518K Vietnamese legal docs → ~500K chunks (article-aware segmentation)
│  Dataset: th1nhng0/vietnamese-legal-documents
│
Stage 1: TSDAE Domain Adaptation
│  Unsupervised denoising autoencoder on legal chunks
│  Loss: DenoisingAutoEncoderLoss (tie_encoder_decoder)
│
Stage 2: Training Data Preparation
│  507K query-passage pairs from phamson02/large-vi-legal-queries
│  + optional synthetic queries via Qwen3-80B (vLLM)
│
Stage 3: Contrastive Fine-tuning R1
│  MatryoshkaLoss(MultipleNegativesRankingLoss) at dims [1024,512,256,128]
│  → NDCG@10 = 0.6725
│
Stage 4: Hard Negative Mining + R2
│  FAISS retrieval → mine rank 50-100 as hard negatives
│  Retrain with triplets (query, positive, hard_negative)
│  → NDCG@10 = 0.6565
│
Stage 5: Multi-task Blending
│  70% retrieval + 20% legal-type classification + 10% STS
│  → NDCG@10 = 0.7229 (best, +5.04 points over R1)
│
Stage 6: Evaluation (MTEB ZacLegalTextRetrieval)
│
Stage 7: Export (HuggingFace Hub + ONNX)
```

**Key insight**: Multi-task blending (Stage 5) provides the largest gain (+5.04 NDCG points), combining retrieval with legal document classification and semantic similarity to prevent overfitting.

## Running the Pipeline

```bash
# Stage 0: Chunk legal documents
python scripts/prepare_data.py

# Stage 1: TSDAE domain adaptation
accelerate launch --num_processes=4 scripts/train_tsdae.py

# Stage 2: Prepare training pairs
python scripts/prepare_training.py

# Stage 2.4 (optional): Generate synthetic queries
python scripts/generate_queries.py

# Stage 3: Contrastive fine-tuning R1
accelerate launch --num_processes=4 scripts/train_contrastive.py

# Stage 4: Mine hard negatives + train R2
python scripts/mine_hard_negatives.py
accelerate launch --num_processes=4 scripts/train_hard_neg.py

# Stage 5: Multi-task blending
accelerate launch --num_processes=4 scripts/train_multitask.py

# Stage 6: Evaluate
python scripts/evaluate.py

# Stage 7: Export to HF Hub + ONNX
python scripts/export.py
```

## Data

| Dataset | Stage | Records | Description |
|---------|-------|---------|-------------|
| `chunks.parquet` | 0 | ~500K | Legal doc chunks with metadata |
| `train.parquet` | 2 | ~480K | Query-passage pairs (with prefixes) |
| `val.parquet` | 2 | ~12.5K | Validation split |
| `test.parquet` | 2 | ~12.5K | Test split |
| `hard_negatives.parquet` | 4 | ~1.5M | Mined triplets (193MB) |
| `synthetic_queries.parquet` | 2.4 | — | LLM-generated pairs (66MB) |

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
│   ├── prepare_data.py      # Stage 0: Chunking
│   ├── train_tsdae.py       # Stage 1: Domain adaptation
│   ├── prepare_training.py  # Stage 2: Data splits
│   ├── generate_queries.py  # Stage 2.4: Synthetic queries
│   ├── train_contrastive.py # Stage 3: Contrastive R1
│   ├── mine_hard_negatives.py # Stage 4: Hard neg mining
│   ├── train_hard_neg.py    # Stage 4: Contrastive R2
│   ├── train_multitask.py   # Stage 5: Multi-task
│   ├── evaluate.py          # Stage 6: MTEB eval
│   ├── export.py            # Stage 7: Export
│   └── utils.py             # Shared utilities
├── data/                    # Training data (parquet)
├── models/                  # Model checkpoints
├── eval/                    # Evaluation results
└── logs/                    # Training logs
```
