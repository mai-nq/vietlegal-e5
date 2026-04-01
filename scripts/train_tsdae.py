"""Stage 1 — TSDAE Domain-Adaptive Pre-training.

Adapts multilingual-e5-large to Vietnamese legal vocabulary using
denoising auto-encoder objective. DenoisingAutoEncoderLoss handles
corruption internally — dataset provides clean text only.

Launch: accelerate launch --num_processes=4 scripts/train_tsdae.py
"""

from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from transformers import EarlyStoppingCallback

from scripts.utils import load_config, setup_logging, ensure_dirs, init_wandb


def main():
    cfg = load_config()
    logger = setup_logging("train_tsdae")
    ensure_dirs(cfg)
    init_wandb(cfg, "tsdae")
    data_dir = Path(cfg["paths"]["data_dir"])

    # Load model
    logger.info(f"Loading base model: {cfg['tsdae']['base_model']}")
    model = SentenceTransformer(cfg["tsdae"]["base_model"])
    model.max_seq_length = cfg["tsdae"]["max_seq_len"]

    # Load chunks — DenoisingAutoEncoderLoss expects two columns:
    # sentence_0 = input (will be corrupted internally by the loss)
    # sentence_1 = target (clean text for reconstruction)
    # Both columns contain the same clean text.
    # Use Arrow-backed Dataset to avoid duplicating 5M strings in RAM per process.
    logger.info("Loading chunks for TSDAE training...")
    dataset = Dataset.from_parquet(str(data_dir / "chunks.parquet"))
    dataset = dataset.rename_column("text", "sentence_0")
    dataset = dataset.add_column("sentence_1", dataset["sentence_0"])
    # Remove non-text columns to save memory
    keep_cols = {"sentence_0", "sentence_1"}
    remove_cols = [c for c in dataset.column_names if c not in keep_cols]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)
    logger.info(f"Total chunks: {len(dataset):,}")

    # Split into train/eval for early stopping
    split = dataset.train_test_split(test_size=0.005, seed=cfg["hardware"]["seed"])
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,}")

    # Loss — tie_encoder_decoder uses XLM-R backbone as both encoder and decoder
    loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

    # Training arguments — batch=8 + grad_accum=8 for effective batch=64 per device
    # Lower per-device batch with higher accumulation to stay within VRAM
    # while increasing effective batch for more stable gradients
    args = SentenceTransformerTrainingArguments(
        output_dir=cfg["tsdae"]["output_dir"],
        num_train_epochs=cfg["tsdae"]["epochs"],
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=cfg["tsdae"]["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=cfg["tsdae"]["warmup_steps"],
        weight_decay=cfg["tsdae"]["weight_decay"],
        bf16=cfg["hardware"]["bf16"],
        gradient_checkpointing=True,
        dataloader_num_workers=cfg["hardware"]["dataloader_workers"],
        save_total_limit=3,
        save_steps=cfg["tsdae"]["save_steps"],
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=cfg["tsdae"]["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=cfg["hardware"]["seed"],
        report_to="wandb",
    )

    # Train with early stopping (patience=3 → stop if eval_loss doesn't improve for 3 evals)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    model.save(cfg["tsdae"]["output_dir"])
    logger.info(f"TSDAE model saved to {cfg['tsdae']['output_dir']}")


if __name__ == "__main__":
    main()
