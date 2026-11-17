#!/usr/bin/env python3
"""Generate local text embeddings with BAAI/bge-m3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

MODEL_NAME = "BAAI/bge-m3"


def choose_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(device: str = "auto"):
    """Load the model once and reuse it for multiple embedding calls."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device=choose_device(device))


def embed_texts(model, texts: Sequence[str]):
    """Return normalized dense vectors for the supplied texts."""
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Provide at least one non-empty text.")
    return model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="one text string to embed")
    source.add_argument("--file", type=Path, help="UTF-8 text file to embed as one item")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, help="write JSON output to this path instead of stdout")
    args = parser.parse_args()

    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    device = choose_device(args.device)
    model = load_model(device)
    vector = embed_texts(model, [text])[0]
    result = {
        "model": MODEL_NAME,
        "device": device,
        "dimensions": int(vector.shape[0]),
        "embedding": vector.tolist(),
    }
    output = json.dumps(result, ensure_ascii=False)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote {result['dimensions']}-dimensional embedding to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
