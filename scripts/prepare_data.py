"""Stage 0 — Data Preparation: Load legal docs, chunk, filter, save parquet."""

import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from scripts.utils import load_config, setup_logging, ensure_dirs

# Approximate ratio: XLM-R tokens per whitespace word (Vietnamese)
TOKENS_PER_WORD = 1.5


def estimate_tokens(text: str) -> int:
    """Fast token count estimate using word count."""
    return int(len(text.split()) * TOKENS_PER_WORD)


def chunk_document(text: str, tokenizer, cfg: dict) -> list[str]:
    """Chunk a legal document by Điều boundaries, with sliding window fallback."""
    max_tokens = cfg["data"]["chunk_max_tokens"]
    window_tokens = cfg["data"]["chunk_window_tokens"]
    overlap_tokens = cfg["data"]["chunk_overlap_tokens"]
    min_chars = cfg["data"]["chunk_min_tokens"] * 3  # ~3 chars per token for Vietnamese

    # Try regex split by Điều (Article) boundaries
    parts = re.split(r"(Điều\s+\d+[^.]*\.)", text)

    chunks = []
    if len(parts) > 1:
        i = 0
        if parts[0].strip():
            chunks.append(parts[0].strip())
            i = 1
        else:
            i = 1
        while i < len(parts) - 1:
            header = parts[i]
            body = parts[i + 1]
            chunks.append((header + body).strip())
            i += 2
        if i < len(parts) and parts[i].strip():
            chunks.append(parts[i].strip())
    else:
        chunks = [text.strip()]

    # Apply sliding window to oversized chunks (only tokenize when needed)
    final_chunks = []
    for chunk in chunks:
        if estimate_tokens(chunk) > max_tokens:
            # Actually tokenize for precise sliding window
            token_ids = tokenizer.encode(chunk, add_special_tokens=False)
            if len(token_ids) > max_tokens:
                start = 0
                while start < len(token_ids):
                    end = start + window_tokens
                    window_ids = token_ids[start:end]
                    window_text = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
                    if window_text:
                        final_chunks.append(window_text)
                    start += window_tokens - overlap_tokens
            else:
                final_chunks.append(chunk)
        else:
            if chunk.strip():
                final_chunks.append(chunk)

    # Merge small chunks (use char length as fast proxy)
    merged = []
    buffer = ""
    for chunk in final_chunks:
        if buffer:
            combined = buffer + " " + chunk
            buffer = ""
            if len(combined) < min_chars:
                buffer = combined
            else:
                merged.append(combined)
        else:
            if len(chunk) < min_chars:
                buffer = chunk
            else:
                merged.append(chunk)
    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    return merged if merged else [text.strip()]


def main():
    cfg = load_config()
    logger = setup_logging("prepare_data")
    ensure_dirs(cfg)
    data_dir = Path(cfg["paths"]["data_dir"])

    # Load tokenizer for token counting (only used for sliding window)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["tsdae"]["base_model"])

    # Load both configs from the legal documents dataset
    logger.info(f"Loading dataset: {cfg['data']['legal_docs_dataset']}")
    ds_metadata = load_dataset(cfg["data"]["legal_docs_dataset"], "metadata", split="data")
    ds_content = load_dataset(cfg["data"]["legal_docs_dataset"], "content", split="data")

    df_meta = ds_metadata.to_pandas()
    df_content = ds_content.to_pandas()
    logger.info(f"Loaded {len(df_meta)} metadata rows, {len(df_content)} content rows")

    # Join on id
    df = df_meta.merge(df_content, on="id", how="inner")
    logger.info(f"After join: {len(df)} docs")

    # Filter: remove short docs
    df = df[df["content"].str.len() >= cfg["data"]["min_content_chars"]]
    logger.info(f"After content length filter: {len(df)} docs")

    # Deduplicate by document_number
    if "document_number" in df.columns:
        df = df.drop_duplicates(subset="document_number", keep="first")
        logger.info(f"After dedup by document_number: {len(df)} docs")

    # Chunk all documents
    logger.info("Chunking documents...")
    all_chunks = []
    doc_count = 0
    for idx, row in df.iterrows():
        chunks = chunk_document(row["content"], tokenizer, cfg)
        for chunk_idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{row['id']}_{chunk_idx}",
                "text": chunk_text,
                "source_doc_id": row["id"],
                "legal_type": row.get("legal_type", ""),
                "legal_sectors": row.get("legal_sectors", ""),
                "chunk_index": chunk_idx,
            })

        doc_count += 1
        if doc_count % 10000 == 0:
            logger.info(f"  {doc_count:,}/{len(df):,} docs → {len(all_chunks):,} chunks")

    chunks_df = pd.DataFrame(all_chunks)
    logger.info(f"Total chunks: {len(chunks_df):,}")

    # Save
    output_path = data_dir / "chunks.parquet"
    chunks_df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    # Stats
    logger.info(f"Text length stats:\n{chunks_df['text'].str.len().describe()}")


if __name__ == "__main__":
    main()
