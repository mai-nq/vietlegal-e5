"""Stage 4.1-4.2 — Hard Negative Mining.

Encodes corpus with R1 model, builds FAISS index, mines hard negatives
from rank 50-100, optionally filters with cross-encoder.
"""

import random
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

from scripts.utils import load_config, setup_logging, ensure_dirs


def main():
    cfg = load_config()
    logger = setup_logging("mine_hard_negatives")
    ensure_dirs(cfg)
    data_dir = Path(cfg["paths"]["data_dir"])

    # Load R1 model
    r1_path = cfg["contrastive_r1"]["output_dir"]
    logger.info(f"Loading R1 model: {r1_path}")
    model = SentenceTransformer(r1_path)

    # Load training data
    train_df = pd.read_parquet(data_dir / "train.parquet")
    logger.info(f"Training data: {len(train_df):,} pairs")

    # Collect unique passages and build query-to-positives mapping
    passages = train_df["positive"].unique().tolist()
    queries = train_df["query"].tolist()
    query_to_positives: dict[str, set[str]] = {}
    for q, p in zip(train_df["query"], train_df["positive"]):
        query_to_positives.setdefault(q, set()).add(p)
    logger.info(f"Unique passages: {len(passages):,}")

    # Create passage-to-index mapping
    passage_to_idx = {p: i for i, p in enumerate(passages)}

    # Encode passages
    logger.info("Encoding passages...")
    passage_embeddings = model.encode(
        passages,
        batch_size=cfg["hard_neg"]["encode_batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    passage_embeddings = np.array(passage_embeddings, dtype=np.float32)

    # Build FAISS index
    dim = passage_embeddings.shape[1]
    logger.info(f"Building FAISS index (dim={dim}, type={cfg['hard_neg']['faiss_index_type']})")
    if cfg["hard_neg"]["faiss_index_type"] == "ivf" and len(passages) > 5_000_000:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, cfg["hard_neg"]["faiss_ivf_nlist"])
        index.train(passage_embeddings)
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(passage_embeddings)
    logger.info("FAISS index built")

    # Load cross-encoder if configured
    reranker = None
    if cfg["hard_neg"]["use_cross_encoder"]:
        logger.info(f"Loading cross-encoder: {cfg['hard_neg']['cross_encoder_model']}")
        reranker = CrossEncoder(cfg["hard_neg"]["cross_encoder_model"])

    # Mine hard negatives in batches
    logger.info("Mining hard negatives...")
    encode_batch = cfg["hard_neg"]["encode_batch_size"]
    top_k = cfg["hard_neg"]["top_k"]
    neg_start = cfg["hard_neg"]["neg_range_start"]
    neg_end = cfg["hard_neg"]["neg_range_end"]
    threshold = cfg["hard_neg"]["cross_encoder_threshold"]
    max_negs = cfg["hard_neg"]["max_negatives_per_query"]

    triplets = []
    unique_queries = list(set(queries))

    # Sample queries to keep mining tractable (cross-encoder is the bottleneck)
    max_queries = cfg["hard_neg"].get("max_queries", len(unique_queries))
    if len(unique_queries) > max_queries:
        random.seed(42)
        random.shuffle(unique_queries)
        unique_queries = unique_queries[:max_queries]
        logger.info(f"Sampled {max_queries:,} queries for mining")

    for batch_start in range(0, len(unique_queries), encode_batch):
        batch_end = min(batch_start + encode_batch, len(unique_queries))
        batch_queries = unique_queries[batch_start:batch_end]

        # Encode query batch
        query_embeddings = model.encode(
            batch_queries,
            batch_size=encode_batch,
            normalize_embeddings=True,
        )
        query_embeddings = np.array(query_embeddings, dtype=np.float32)

        # Search FAISS
        scores, indices = index.search(query_embeddings, top_k)

        for q_idx, (query, retrieved_indices) in enumerate(zip(batch_queries, indices)):
            positives = query_to_positives.get(query, set())
            positive_indices = {passage_to_idx.get(p, -1) for p in positives}

            # Get candidates from rank neg_start to neg_end, excluding all true positives
            candidates = []
            for rank_idx in range(neg_start, min(neg_end, len(retrieved_indices))):
                p_idx = retrieved_indices[rank_idx]
                if p_idx in positive_indices or p_idx < 0:
                    continue
                candidates.append(passages[p_idx])

            if not candidates:
                continue

            # Cross-encoder filtering
            if reranker is not None and candidates:
                raw_query = query.replace("query: ", "", 1)
                raw_candidates = [c.replace("passage: ", "", 1) for c in candidates]
                pairs = [[raw_query, c] for c in raw_candidates]
                ce_scores = reranker.predict(pairs, batch_size=512)

                # REMOVE candidates with score > threshold (likely true positives)
                # KEEP candidates with score <= threshold (genuine hard negatives)
                filtered = [
                    (cand, score)
                    for cand, score in zip(candidates, ce_scores)
                    if score <= threshold
                ]
                # Sort by descending score (hardest remaining negatives)
                filtered.sort(key=lambda x: x[1], reverse=True)
                hard_negs = [cand for cand, _ in filtered[:max_negs]]
            else:
                hard_negs = candidates[:max_negs]

            # Use first positive for triplet (all positives are equivalent for this query)
            primary_positive = next(iter(positives)) if positives else ""
            for neg in hard_negs:
                triplets.append({
                    "query": query,
                    "positive": primary_positive,
                    "hard_negative": neg,
                })

        if batch_start % (encode_batch * 10) == 0:
            logger.info(f"  Processed {batch_end:,}/{len(unique_queries):,} queries, {len(triplets):,} triplets")

    # Save
    triplets_df = pd.DataFrame(triplets)
    output_path = data_dir / "hard_negatives.parquet"
    triplets_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(triplets_df):,} triplets to {output_path}")


if __name__ == "__main__":
    main()
