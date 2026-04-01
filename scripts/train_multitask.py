"""Stage 5 — Multi-task Blending.

Combines retrieval, classification (by legal_type), and STS datasets.
Uses ST v3 dict-of-datasets + dict-of-losses API with PROPORTIONAL sampling.

Launch: accelerate launch --num_processes=4 scripts/train_multitask.py
"""

from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import (
    CoSENTLoss,
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)

from scripts.utils import (
    load_config,
    setup_logging,
    ensure_dirs,
    init_wandb,
    add_passage_prefix,
)


def build_classification_dataset(chunks_df, target_size, logger):
    """Build classification pairs from chunks with same legal_type."""
    pairs = []
    for legal_type, group in chunks_df.groupby("legal_type"):
        texts = group["text"].tolist()
        if len(texts) < 2:
            continue
        for i in range(0, len(texts) - 1, 2):
            pairs.append({
                "sentence_0": add_passage_prefix(texts[i]),
                "sentence_1": add_passage_prefix(texts[i + 1]),
            })

    df = pd.DataFrame(pairs)
    if len(df) == 0:
        return None

    # Adjust to target size
    if len(df) > target_size:
        df = df.sample(n=target_size, random_state=42)
    elif len(df) < target_size:
        # Oversample
        repeats = target_size // len(df) + 1
        df = pd.concat([df] * repeats, ignore_index=True).head(target_size)

    logger.info(f"Classification dataset: {len(df):,} pairs")
    return Dataset.from_dict({
        "sentence_0": df["sentence_0"].tolist(),
        "sentence_1": df["sentence_1"].tolist(),
    })


def build_sts_dataset(target_size, logger):
    """Try loading Vietnamese STS dataset."""
    try:
        sts_ds = load_dataset("stsb_multi_mt", "vi", split="train")
        df = pd.DataFrame({
            "sentence_0": [add_passage_prefix(r["sentence1"]) for r in sts_ds],
            "sentence_1": [add_passage_prefix(r["sentence2"]) for r in sts_ds],
            "score": [r["similarity_score"] / 5.0 for r in sts_ds],
        })

        # Adjust to target size
        if len(df) > target_size:
            df = df.sample(n=target_size, random_state=42)
        elif len(df) < target_size:
            repeats = target_size // len(df) + 1
            df = pd.concat([df] * repeats, ignore_index=True).head(target_size)

        logger.info(f"STS dataset: {len(df):,} pairs")
        return Dataset.from_dict({
            "sentence_0": df["sentence_0"].tolist(),
            "sentence_1": df["sentence_1"].tolist(),
            "score": df["score"].tolist(),
        })
    except Exception as e:
        logger.warning(f"Could not load STS dataset: {e}")
        return None


def build_evaluator(val_df, train_df, logger):
    """Build IR evaluator (same as R1/R2)."""
    queries = {}
    corpus = {}
    relevant_docs = {}

    for i, row in val_df.iterrows():
        qid = f"q{i}"
        cid = f"c{i}"
        queries[qid] = row["query"]
        corpus[cid] = row["positive"]
        relevant_docs[qid] = {cid: 1}

    train_sample = train_df.sample(n=min(10000, len(train_df)), random_state=42)
    for j, row in enumerate(train_sample.itertuples()):
        corpus[f"d{j}"] = row.positive

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="val",
        ndcg_at_k=[10],
        show_progress_bar=True,
    )


def main():
    cfg = load_config()
    logger = setup_logging("train_multitask")
    ensure_dirs(cfg)
    init_wandb(cfg, "multitask")
    data_dir = Path(cfg["paths"]["data_dir"])
    dims = cfg["contrastive_r1"]["matryoshka_dims"]

    # Load R2 model
    r2_path = cfg["contrastive_r2"]["output_dir"]
    logger.info(f"Loading R2 model: {r2_path}")
    model = SentenceTransformer(r2_path)

    # Build retrieval dataset
    train_df = pd.read_parquet(data_dir / "train.parquet")
    retrieval_ds = Dataset.from_dict({
        "sentence_0": train_df["query"].tolist(),
        "sentence_1": train_df["positive"].tolist(),
    })
    retrieval_size = len(retrieval_ds)

    # Calculate target sizes based on sampling ratios
    ratios = cfg["multitask"]["sampling_ratios"]

    # Build STS dataset first to check availability
    target_total = int(retrieval_size / ratios["retrieval"])
    sts_ds = build_sts_dataset(int(target_total * ratios["sts"]), logger)

    # Adjust ratios if no STS
    if sts_ds is None:
        logger.info("No STS dataset — redistributing: 80% retrieval, 20% classification")
        target_class = int(retrieval_size * 0.25)  # 20/80 = 0.25 of retrieval
    else:
        target_class = int(target_total * ratios["classification"])

    # Build classification dataset
    chunks_df = pd.read_parquet(data_dir / "chunks.parquet")
    classification_ds = build_classification_dataset(chunks_df, target_class, logger)

    # Assemble dataset and loss dicts
    train_datasets = {"retrieval": retrieval_ds}
    losses = {
        "retrieval": MatryoshkaLoss(
            model, MultipleNegativesRankingLoss(model), matryoshka_dims=dims
        ),
    }

    if classification_ds is not None:
        train_datasets["classification"] = classification_ds
        losses["classification"] = MatryoshkaLoss(
            model, MultipleNegativesRankingLoss(model), matryoshka_dims=dims
        )

    if sts_ds is not None:
        train_datasets["sts"] = sts_ds
        losses["sts"] = MatryoshkaLoss(
            model, CoSENTLoss(model), matryoshka_dims=dims
        )

    logger.info(f"Training with datasets: {list(train_datasets.keys())}")

    # Evaluator
    val_df = pd.read_parquet(data_dir / "val.parquet")
    evaluator = build_evaluator(val_df, train_df, logger)

    # Training args
    args = SentenceTransformerTrainingArguments(
        output_dir=cfg["multitask"]["output_dir"],
        num_train_epochs=cfg["multitask"]["epochs"],
        learning_rate=cfg["multitask"]["lr"],
        per_device_train_batch_size=cfg["contrastive_r1"]["per_device_batch_size"],
        gradient_accumulation_steps=cfg["contrastive_r1"]["grad_accum_steps"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=cfg["hardware"]["bf16"],
        multi_dataset_batch_sampler="proportional",
        eval_steps=cfg["contrastive_r1"]["eval_steps"],
        save_steps=cfg["contrastive_r1"]["eval_steps"],
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_cosine_ndcg@10",
        dataloader_num_workers=cfg["hardware"]["dataloader_workers"],
        seed=cfg["hardware"]["seed"],
        report_to="wandb",
    )

    # Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_datasets,
        loss=losses,
        evaluator=evaluator,
    )
    trainer.train()
    model.save(cfg["multitask"]["output_dir"])
    logger.info(f"Final model saved to {cfg['multitask']['output_dir']}")


if __name__ == "__main__":
    main()
