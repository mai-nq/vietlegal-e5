"""Stage 2.1 — Training Data Preparation.

Loads query-passage pairs, adds mE5 prefixes, splits into train/val/test.
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from scripts.utils import load_config, setup_logging, ensure_dirs, add_query_prefix, add_passage_prefix


def main():
    cfg = load_config()
    logger = setup_logging("prepare_training")
    ensure_dirs(cfg)
    data_dir = Path(cfg["paths"]["data_dir"])

    # Load query-passage pairs
    logger.info(f"Loading dataset: {cfg['data']['query_pairs_dataset']}")
    ds = load_dataset(cfg["data"]["query_pairs_dataset"], split="train")
    df = ds.to_pandas()
    logger.info(f"Loaded {len(df):,} query-passage pairs")
    logger.info(f"Columns: {list(df.columns)}")

    # Rename context -> positive (dataset uses 'context' for passage text)
    if "positive" not in df.columns and "context" in df.columns:
        df = df.rename(columns={"context": "positive"})

    # Add mE5 prefixes — CRITICAL for model performance
    df["query"] = df["query"].apply(add_query_prefix)
    df["positive"] = df["positive"].apply(add_passage_prefix)

    # Split: 95% train, 2.5% val, 2.5% test
    seed = cfg["split"]["seed"]
    train_df, temp_df = train_test_split(df, test_size=0.05, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    # Save
    train_df.to_parquet(data_dir / "train.parquet", index=False)
    val_df.to_parquet(data_dir / "val.parquet", index=False)
    test_df.to_parquet(data_dir / "test.parquet", index=False)

    logger.info(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    logger.info(f"Query prefix check: {train_df['query'].iloc[0][:30]}")
    logger.info(f"Passage prefix check: {train_df['positive'].iloc[0][:30]}")


if __name__ == "__main__":
    main()
