"""Stage 7 — Export & Deploy.

Push best model to HuggingFace Hub, export ONNX with optional FP16 quantization.
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer

from scripts.utils import load_config, setup_logging, ensure_dirs


MODEL_CARD_TEMPLATE = """---
language: vi
tags:
- sentence-transformers
- embedding
- legal
- vietnamese
- matryoshka
license: apache-2.0
---

# VietLegal-E5

Vietnamese legal domain embedding model fine-tuned from `intfloat/multilingual-e5-large`.

## Usage

**IMPORTANT**: This model requires prefixes:
- Queries: `"query: "` + your query
- Passages/Documents: `"passage: "` + your text

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{hub_model_name}")
queries = model.encode(["query: quyền sở hữu trí tuệ là gì?"])
passages = model.encode(["passage: Điều 1. Luật Sở hữu trí tuệ..."])
```

## Matryoshka Dimensions

Supports truncation to: 1024, 512, 256, 128 dimensions.

## Training Pipeline

1. TSDAE domain adaptation on 518K Vietnamese legal documents
2. Contrastive fine-tuning with MatryoshkaLoss on ~500K query-passage pairs
3. Hard negative mining with FAISS + cross-encoder filtering
4. Round 2 contrastive training with hard negatives
5. Multi-task blending (retrieval + classification + STS)

## Benchmarks

See eval/results.json for full comparison on Zalo AI 2021 Legal Text Retrieval.
"""


def main():
    cfg = load_config()
    logger = setup_logging("export")
    ensure_dirs(cfg)

    # Determine best model path
    best_model_path = cfg["multitask"]["output_dir"]
    if not Path(best_model_path).exists():
        # Fallback to R2, then R1
        for fallback in [cfg["contrastive_r2"]["output_dir"], cfg["contrastive_r1"]["output_dir"]]:
            if Path(fallback).exists():
                best_model_path = fallback
                break
    logger.info(f"Best model path: {best_model_path}")

    # Load model
    model = SentenceTransformer(best_model_path)

    # Write model card before pushing
    hub_name = cfg["export"]["hub_model_name"]
    model_card_content = MODEL_CARD_TEMPLATE.format(hub_model_name=hub_name)
    readme_path = Path(best_model_path) / "README.md"
    readme_path.write_text(model_card_content, encoding="utf-8")
    logger.info(f"Model card written to {readme_path}")

    # Push to HuggingFace Hub
    logger.info(f"Pushing model to HuggingFace Hub: {hub_name}")
    model.push_to_hub(hub_name, private=False)
    logger.info(f"Model pushed to Hub successfully")

    # ONNX export
    if cfg["export"]["onnx_export"]:
        logger.info("Exporting to ONNX...")
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        onnx_path = Path(cfg["paths"]["model_dir"]) / "vietlegal-e5-onnx"

        # Export
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            best_model_path, export=True
        )
        tokenizer = AutoTokenizer.from_pretrained(best_model_path)
        ort_model.save_pretrained(onnx_path)
        tokenizer.save_pretrained(onnx_path)
        logger.info(f"ONNX model exported to {onnx_path}")

        # FP16 quantization
        if cfg["export"]["onnx_quantize_fp16"]:
            logger.info("Applying FP16 quantization...")
            import onnx
            from onnxruntime.transformers.float16 import convert_float_to_float16

            onnx_model_path = onnx_path / "model.onnx"
            model_fp16 = onnx.load(str(onnx_model_path))
            model_fp16 = convert_float_to_float16(model_fp16)
            fp16_path = onnx_path / "model_fp16.onnx"
            onnx.save(model_fp16, str(fp16_path))
            logger.info(f"FP16 ONNX model saved to {fp16_path}")

        # Validate ONNX
        logger.info("Validating ONNX export...")
        ort_model = ORTModelForFeatureExtraction.from_pretrained(str(onnx_path))
        tokenizer = AutoTokenizer.from_pretrained(str(onnx_path))
        inputs = tokenizer("passage: Điều 1. Phạm vi điều chỉnh", return_tensors="pt")
        outputs = ort_model(**inputs)
        logger.info(f"ONNX output shape: {outputs.last_hidden_state.shape}")

    logger.info("Export complete!")
    logger.info(f"TEI compatibility: verify manually with:")
    logger.info(f"  docker run --gpus all -p 8080:80 -v $(pwd)/models/vietlegal-e5-onnx:/model \\")
    logger.info(f"    ghcr.io/huggingface/text-embeddings-inference:latest --model-id /model")


if __name__ == "__main__":
    main()
