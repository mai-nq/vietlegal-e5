"""Stage 2.4 — Synthetic Query Generation via vLLM + Qwen3.

Launches vLLM server, generates legal questions for each chunk,
saves as query-passage pairs, merges into training set.
"""

import asyncio
import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests

from scripts.utils import (
    load_config,
    setup_logging,
    ensure_dirs,
    add_query_prefix,
    add_passage_prefix,
)

PROMPT_TEMPLATE = """Bạn là chuyên gia pháp luật Việt Nam. Dựa trên đoạn văn bản pháp luật sau,
hãy viết đúng 3 câu hỏi pháp lý mà đoạn văn này có thể trả lời.
Câu hỏi phải tự nhiên, đa dạng về cách hỏi.
Mỗi câu hỏi trên một dòng, đánh số 1., 2., 3.

Đoạn văn: {chunk_text}"""


def launch_vllm_server(cfg, logger, model_name):
    """Launch vLLM OpenAI-compatible server as subprocess."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--tensor-parallel-size", str(cfg["synth"]["vllm_tensor_parallel"]),
        "--max-model-len", str(cfg["synth"]["vllm_max_model_len"]),
        "--port", str(cfg["synth"]["vllm_port"]),
        "--dtype", "auto",
    ]
    logger.info(f"Launching vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def wait_for_health(port, timeout_iters=60, sleep_sec=10):
    """Poll vLLM health endpoint, return True if healthy."""
    for _ in range(timeout_iters):
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=5)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(sleep_sec)
    return False


def parse_queries(response_text: str) -> list[str]:
    """Extract numbered queries from LLM response."""
    queries = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        match = re.match(r"^\d+\.\s*(.+)$", line)
        if match:
            queries.append(match.group(1).strip())
    return queries


async def generate_batch(client, model_name, chunks, cfg, logger):
    """Generate queries for a batch of chunks concurrently."""
    async def gen_one(chunk_text):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(chunk_text=chunk_text)}],
                max_tokens=cfg["synth"]["max_new_tokens"],
                temperature=cfg["synth"]["temperature"],
            )
            return chunk_text, response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return chunk_text, ""

    tasks = [gen_one(chunk) for chunk in chunks]
    return await asyncio.gather(*tasks)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None, help="Override vLLM port")
    parser.add_argument("--external-vllm", action="store_true",
                        help="Use an already-running vLLM server (skip launch/shutdown)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model name for API calls")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Stop after generating this many pairs")
    args = parser.parse_args()

    cfg = load_config()
    logger = setup_logging("generate_queries")
    ensure_dirs(cfg)
    data_dir = Path(cfg["paths"]["data_dir"])
    port = args.port or cfg["synth"]["vllm_port"]
    model_name = args.model_name or cfg["synth"]["model_name"]

    # Load chunks
    logger.info("Loading chunks...")
    chunks_df = pd.read_parquet(data_dir / "chunks.parquet")
    all_chunks = chunks_df["text"].tolist()

    # Load checkpoint if exists
    checkpoint_path = data_dir / "synth_checkpoint.json"
    start_idx = 0
    accumulated_pairs = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_idx = ckpt["last_index"]
        logger.info(f"Resuming from checkpoint at index {start_idx}")
        # Load existing synthetic pairs
        synth_path = data_dir / "synthetic_queries.parquet"
        if synth_path.exists():
            existing = pd.read_parquet(synth_path)
            accumulated_pairs = existing.to_dict("records")

    server_proc = None
    if args.external_vllm:
        logger.info(f"Using external vLLM server on port {port}")
        healthy = wait_for_health(port, timeout_iters=3, sleep_sec=2)
        if not healthy:
            raise RuntimeError(f"External vLLM server not reachable on port {port}")
        logger.info("External vLLM server is healthy!")
    else:
        # Launch vLLM server
        server_proc = launch_vllm_server(cfg, logger, model_name)

        logger.info(f"Waiting for vLLM server on port {port}...")
        healthy = wait_for_health(port)

        if not healthy:
            logger.warning("Primary model failed. Trying fallback FP8 model...")
            server_proc.terminate()
            server_proc.wait()
            model_name = cfg["synth"]["fallback_model"]
            server_proc = launch_vllm_server(cfg, logger, model_name)
            healthy = wait_for_health(port)
            if not healthy:
                server_proc.terminate()
                raise RuntimeError("Both primary and fallback models failed to load in vLLM")

        logger.info("vLLM server is healthy!")

    try:
        from openai import AsyncOpenAI

        batch_size = cfg["synth"]["batch_size"]
        checkpoint_every = cfg["synth"]["checkpoint_every_n"]

        async def run_all_batches():
            client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")

            for batch_start in range(start_idx, len(all_chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(all_chunks))
                batch_chunks = all_chunks[batch_start:batch_end]

                results = await generate_batch(client, model_name, batch_chunks, cfg, logger)

                for chunk_text, response_text in results:
                    if not response_text:
                        continue
                    queries = parse_queries(response_text)
                    for q in queries:
                        accumulated_pairs.append({
                            "query": add_query_prefix(q),
                            "positive": add_passage_prefix(chunk_text),
                        })

                # Checkpoint
                if (batch_end % checkpoint_every < batch_size) or batch_end == len(all_chunks):
                    synth_df = pd.DataFrame(accumulated_pairs)
                    synth_df.to_parquet(data_dir / "synthetic_queries.parquet", index=False)
                    with open(checkpoint_path, "w") as f:
                        json.dump({"last_index": batch_end}, f)
                    logger.info(f"Checkpoint at {batch_end:,}/{len(all_chunks):,} — {len(accumulated_pairs):,} pairs")

                # Early stop if target reached
                if args.max_pairs and len(accumulated_pairs) >= args.max_pairs:
                    logger.info(f"Reached target {args.max_pairs:,} pairs. Stopping.")
                    return

        asyncio.run(run_all_batches())

        # Final save
        synth_df = pd.DataFrame(accumulated_pairs)
        synth_df.to_parquet(data_dir / "synthetic_queries.parquet", index=False)
        logger.info(f"Total synthetic pairs: {len(synth_df):,}")

        # Merge into train only (val/test untouched)
        train_df = pd.read_parquet(data_dir / "train.parquet")
        merged = pd.concat([train_df, synth_df], ignore_index=True)
        merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
        merged.to_parquet(data_dir / "train.parquet", index=False)
        logger.info(f"Updated train.parquet: {len(merged):,} total pairs")

        # Cleanup checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    finally:
        if server_proc is not None:
            logger.info("Shutting down vLLM server...")
            server_proc.terminate()
            server_proc.wait()
        else:
            logger.info("External vLLM server left running.")


if __name__ == "__main__":
    main()
