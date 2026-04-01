"""Stage 6 — Evaluation on Zalo Legal benchmark via MTEB.

Evaluates all checkpoints and baselines using the standard MTEB
ZacLegalTextRetrieval task (GreenNode/zalo-ai-legal-text-retrieval-vn).
"""

import json
from pathlib import Path

import mteb
import torch
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from sentence_transformers import SentenceTransformer

from scripts.utils import load_config, setup_logging, ensure_dirs, init_wandb


def _needs_e5_prefix(model_name: str) -> bool:
    """Check if model uses e5-style 'query: '/'passage: ' prefixes."""
    return "e5" in model_name.lower()


def main():
    cfg = load_config()
    logger = setup_logging("evaluate")
    ensure_dirs(cfg)
    init_wandb(cfg, "evaluation")
    eval_dir = Path(cfg["paths"]["eval_dir"])

    # Define models to evaluate
    models_to_eval = {}

    # Our checkpoints
    if cfg["eval"]["eval_all_checkpoints"]:
        for name, path in [
            ("vietlegal-e5-tsdae", cfg["tsdae"]["output_dir"]),
            ("vietlegal-e5-r1", cfg["contrastive_r1"]["output_dir"]),
            ("vietlegal-e5-r2", cfg["contrastive_r2"]["output_dir"]),
            ("vietlegal-e5-final", cfg["multitask"]["output_dir"]),
        ]:
            if Path(path).exists():
                models_to_eval[name] = path

    # Baselines
    for baseline in cfg["eval"]["baselines"]:
        models_to_eval[baseline.split("/")[-1]] = baseline

    logger.info(f"Models to evaluate: {list(models_to_eval.keys())}")

    all_results = {}
    for model_name, model_path in models_to_eval.items():
        logger.info(f"Evaluating: {model_name}")

        try:
            st_model = SentenceTransformer(model_path)
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            continue

        # e5 models need query/passage prompts
        if _needs_e5_prefix(model_name):
            model_prompts = {"query": "query: ", "document": "passage: "}
        else:
            model_prompts = {}

        # Determine dims to evaluate
        if "halong" in model_name:
            dims = cfg["eval"]["halong_eval_dims"]
        elif model_name.startswith("vietlegal-e5-"):
            dims = cfg["eval"]["matryoshka_dims"]
        else:
            dims = [st_model.get_sentence_embedding_dimension()]

        for dim in dims:
            eval_name = f"{model_name}_dim{dim}"
            logger.info(f"  Evaluating at dim={dim}...")

            st_model.truncate_dim = dim
            wrapper = SentenceTransformerEncoderWrapper(
                model=st_model,
                model_prompts=model_prompts,
            )
            eval_task = mteb.get_task(cfg["eval"]["mteb_task"])
            result = mteb.evaluate(
                model=wrapper,
                tasks=[eval_task],
                overwrite_strategy="always",
            )

            # Extract NDCG@10 from result
            task_result = result.task_results[0]
            scores = task_result.scores.get("test", [{}])
            ndcg = None
            for score_dict in scores:
                if "ndcg_at_10" in score_dict:
                    ndcg = score_dict["ndcg_at_10"]
                    break

            all_results[eval_name] = {"ndcg@10": ndcg, "full_scores": scores}
            logger.info(f"  {eval_name} NDCG@10: {ndcg}")

        del st_model
        torch.cuda.empty_cache()

    # Save results
    output_path = eval_dir / "results.json"
    results_summary = {k: v["ndcg@10"] for k, v in all_results.items()}
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    # Print summary table
    logger.info("\n=== EVALUATION RESULTS (MTEB ZacLegalTextRetrieval) ===")
    logger.info(f"{'Model':<45} {'NDCG@10':>10}")
    logger.info("-" * 57)
    for model_name, ndcg in sorted(results_summary.items()):
        if isinstance(ndcg, (int, float)):
            logger.info(f"{model_name:<45} {ndcg:>10.4f}")
        else:
            logger.info(f"{model_name:<45} {'N/A':>10}")


if __name__ == "__main__":
    main()
