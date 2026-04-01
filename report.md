# VietLegal-E5 Evaluation Report

## Benchmark

**Task**: MTEB ZacLegalTextRetrieval
**Dataset**: [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn)
**Metric**: NDCG@10
**Corpus**: 61,425 Vietnamese legal documents
**Queries**: 818 test queries
**Split**: test

## Results

### Our Models

| Model | Stage | Dim 1024 | Dim 512 | Dim 256 | Dim 128 |
|-------|-------|----------|---------|---------|---------|
| **vietlegal-e5** | Multitask fine-tuning | **0.7229** | **0.7208** | **0.7058** | **0.7073** |
| vietlegal-e5-contrastive | Contrastive learning | 0.6725 | 0.6665 | 0.6575 | 0.6577 |
| vietlegal-e5-hard-neg | Hard negatives contrastive | 0.6565 | 0.6471 | 0.6277 | 0.6379 |

### Baselines

| Model | Params | Dim | NDCG@10 |
|-------|--------|-----|---------|
| microsoft/harrier-oss-v1-0.6b | 600M | 1024 | 0.7210 |
| intfloat/multilingual-e5-large | 560M | 1024 | 0.6660 |
| bkai-foundation-models/vietnamese-bi-encoder | 135M | 768 | 0.6160 |
| intfloat/multilingual-e5-base | 278M | 768 | 0.6030 |
| contextboxai/halong_embedding | 278M | 768 | 0.6009 |
| contextboxai/halong_embedding | 278M | 256 | 0.5494 |
| contextboxai/halong_embedding | 278M | 128 | 0.5100 |

## Key Findings

1. **vietlegal-e5 achieves the best score** (NDCG@10 = 0.7229), beating all baselines including microsoft/harrier-oss-v1-0.6b (0.7210) — a model with more parameters.

2. **Improvement over base model**: +5.69 points over the multilingual-e5-large baseline (0.7229 vs 0.6660), demonstrating the effectiveness of domain-specific fine-tuning for Vietnamese legal text retrieval.

3. **Matryoshka embeddings work well**: vietlegal-e5 at dim=128 (0.7073) still outperforms multilingual-e5-large at dim=1024 (0.6660) — 8x smaller embeddings with better performance.

4. **Hard negatives degraded performance**: The hard negatives stage performed worse than standard contrastive, suggesting the hard negatives mining or training configuration needs revision.

5. **Multitask fine-tuning provides the biggest improvement**: The jump from contrastive (0.6725) to final (0.7229) is +5.04 points, the largest gain in the pipeline.

## Training Pipeline

```
multilingual-e5-large (baseline: 0.6660)
    |
    v
vietlegal-e5-contrastive (contrastive: 0.6725, +0.65)
    |
    v
vietlegal-e5-hard-neg (hard negatives: 0.6565, -1.60)  <-- degraded
    |
    v
vietlegal-e5 (multitask: 0.7229, +6.64)
```

## Evaluation Setup

- All models evaluated using the `mteb` library with `mteb.evaluate()`
- E5-style models use `"query: "` / `"passage: "` prefixes
- Harrier uses instruction-based query prompt
- halong_embedding, vietnamese-bi-encoder: no prefix (raw text input)
- GPU: NVIDIA H100 80GB
