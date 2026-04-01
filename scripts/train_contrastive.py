"""Stage 3 — Contrastive Fine-tuning Round 1.

MatryoshkaLoss wrapping MultipleNegativesRankingLoss at dims [1024, 512, 256, 128].
Uses TSDAE model if available, otherwise falls back to raw mE5-large.

Launch: accelerate launch --num_processes=4 scripts/train_contrastive.py
"""

from pathlib import Path

import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

from scripts.utils import load_config, setup_logging, ensure_dirs, init_wandb, build_evaluator


def main():
    cfg = load_config()
    logger = setup_logging("train_contrastive")
    ensure_dirs(cfg)
    init_wandb(cfg, "contrastive-r1")
    data_dir = Path(cfg["paths"]["data_dir"])

    # Use base model specified in contrastive_r1 config, fallback to raw mE5-large
    base_model = cfg["contrastive_r1"].get("base_model", cfg["tsdae"]["base_model"])
    logger.info(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)

    # Load datasets
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")

    train_ds = Dataset.from_dict({
        "sentence_0": train_df["query"].tolist(),
        "sentence_1": train_df["positive"].tolist(),
    })
    val_ds = Dataset.from_dict({
        "sentence_0": val_df["query"].tolist(),
        "sentence_1": val_df["positive"].tolist(),
    })
    logger.info(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    # Loss: Matryoshka wrapping MNRL
    mnrl = MultipleNegativesRankingLoss(model)
    loss = MatryoshkaLoss(
        model, mnrl, matryoshka_dims=cfg["contrastive_r1"]["matryoshka_dims"]
    )

    # Evaluator
    evaluator = build_evaluator(val_df, train_df, logger)

    # Training arguments
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
        save_steps=cfg["contrastive_r1"]["eval_steps"],
        eval_strategy="steps",
        save_total_limit=cfg["contrastive_r1"]["save_total_limit"],
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
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    model.save(cfg["contrastive_r1"]["output_dir"])
    logger.info(f"R1 model saved to {cfg['contrastive_r1']['output_dir']}")


if __name__ == "__main__":
    main()
