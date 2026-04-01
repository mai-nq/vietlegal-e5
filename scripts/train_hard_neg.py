"""Stage 4.3 — Contrastive Fine-tuning Round 2 with Hard Negatives.

Same loss as R1 but lower LR, uses triplets (query, positive, hard_negative).
MNRL with 3 columns automatically treats column 2 as hard negative.

Launch: accelerate launch --num_processes=4 scripts/train_hard_neg.py
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
    logger = setup_logging("train_hard_neg")
    ensure_dirs(cfg)
    init_wandb(cfg, "contrastive-r2")
    data_dir = Path(cfg["paths"]["data_dir"])

    # Load R1 model as base
    r1_path = cfg["contrastive_r1"]["output_dir"]
    logger.info(f"Loading R1 model: {r1_path}")
    model = SentenceTransformer(r1_path)

    # Load hard negative triplets
    hn_df = pd.read_parquet(data_dir / "hard_negatives.parquet")
    logger.info(f"Hard negative triplets: {len(hn_df):,}")

    # Convert to HF Dataset with sentence_0, sentence_1, sentence_2
    train_ds = Dataset.from_dict({
        "sentence_0": hn_df["query"].tolist(),
        "sentence_1": hn_df["positive"].tolist(),
        "sentence_2": hn_df["hard_negative"].tolist(),
    })

    # Load val for evaluator
    val_df = pd.read_parquet(data_dir / "val.parquet")
    train_df = pd.read_parquet(data_dir / "train.parquet")

    # Loss: same Matryoshka(MNRL) — MNRL handles triplets natively
    mnrl = MultipleNegativesRankingLoss(model)
    dims = cfg["contrastive_r1"]["matryoshka_dims"]  # from R1 config
    loss = MatryoshkaLoss(model, mnrl, matryoshka_dims=dims)

    # Evaluator
    evaluator = build_evaluator(val_df, train_df, logger)

    # Training args — R2-specific lr/epochs, rest inherited from R1
    args = SentenceTransformerTrainingArguments(
        output_dir=cfg["contrastive_r2"]["output_dir"],
        num_train_epochs=cfg["contrastive_r2"]["epochs"],               # 2 (R2)
        learning_rate=cfg["contrastive_r2"]["lr"],                       # 5e-6 (R2)
        per_device_train_batch_size=cfg["contrastive_r1"]["per_device_batch_size"],  # 32 (from R1)
        gradient_accumulation_steps=cfg["contrastive_r1"]["grad_accum_steps"],       # 2 (from R1)
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
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    model.save(cfg["contrastive_r2"]["output_dir"])
    logger.info(f"R2 model saved to {cfg['contrastive_r2']['output_dir']}")


if __name__ == "__main__":
    main()
