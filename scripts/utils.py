"""Shared utilities for VietLegal-E5 pipeline."""

import logging
import os
from pathlib import Path

import yaml
import wandb


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the YAML config as a dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve paths relative to base_dir
    base = Path(cfg["paths"]["base_dir"])
    cfg["paths"]["data_dir"] = str(base / cfg["paths"]["data_dir"])
    cfg["paths"]["model_dir"] = str(base / cfg["paths"]["model_dir"])
    cfg["paths"]["eval_dir"] = str(base / cfg["paths"]["eval_dir"])
    return cfg


def add_prefix(text: str, prefix: str) -> str:
    """Add prefix if not already present."""
    if text.startswith(prefix):
        return text
    return prefix + text


def add_query_prefix(text: str) -> str:
    return add_prefix(text, "query: ")


def add_passage_prefix(text: str) -> str:
    return add_prefix(text, "passage: ")


def init_wandb(cfg: dict, run_name: str):
    """Initialize W&B run from config."""
    wandb.init(
        project=cfg["wandb"]["project"],
        name=run_name,
        tags=cfg["wandb"]["tags"],
        config=cfg,
    )


def setup_logging(name: str) -> logging.Logger:
    """Set up a logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger


def ensure_dirs(cfg: dict):
    """Create output directories if they don't exist."""
    for key in ["data_dir", "model_dir", "eval_dir"]:
        os.makedirs(cfg["paths"][key], exist_ok=True)


def build_evaluator(val_df, train_df, logger):
    """Build IR evaluator with val queries + distractor corpus."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    queries = {}
    corpus = {}
    relevant_docs = {}

    for i, row in val_df.iterrows():
        qid = f"q{i}"
        cid = f"c{i}"
        queries[qid] = row["query"]
        corpus[cid] = row["positive"]
        relevant_docs[qid] = {cid: 1}

    # Add ~10K distractors from train
    train_sample = train_df.sample(n=min(10000, len(train_df)), random_state=42)
    for j, row in enumerate(train_sample.itertuples()):
        corpus[f"d{j}"] = row.positive

    logger.info(f"Evaluator: {len(queries)} queries, {len(corpus)} corpus docs")

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="val",
        ndcg_at_k=[10],
        show_progress_bar=True,
    )
