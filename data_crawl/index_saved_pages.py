#!/usr/bin/env python3
"""Chunk saved Mineralogy text, embed it locally, and upsert it into Qdrant."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from local_embeddings import MODEL_NAME, embed_texts, load_model


ROOT = Path(__file__).resolve().parent.parent
SAVED_PAGES = ROOT / "data" / "opengeology_mineralogy" / "processed" / "saved_pages"
MANIFEST = SAVED_PAGES / "manifest.json"
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "mineralogy_chunks_v1"
VECTOR_SIZE = 1024
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 70
CHUNKING_VERSION = "section-aware-500-overlap-70-v1"
NAMESPACE = uuid.UUID("af718756-3993-4288-9c61-31216409a3a3")


def qdrant_request(method: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        QDRANT_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qdrant {method} {path} returned HTTP {exc.code}: {detail}") from exc


def section_at(text: str, char_pos: int, chapter_title: str) -> str:
    section = chapter_title
    for line in text[:char_pos].splitlines():
        line = line.strip()
        if line == chapter_title:
            section = chapter_title
        elif re.match(r"^\d+(?:\.\d+)+\s+\S", line) and len(line) <= 180:
            section = line
    return section


def sentence_boundary(token_ends: list[int], text: str, char_start: int, char_end: int, min_token: int, max_token: int) -> int | None:
    region = text[char_start:char_end]
    candidates = []
    for match in re.finditer(r"(?<=[.!?])\s+|\n+", region):
        absolute_end = char_start + match.end()
        token_end = bisect.bisect_left(token_ends, absolute_end) + 1
        if min_token <= token_end <= max_token:
            candidates.append(token_end)
    return max(candidates) if candidates else None


def chunk_document(text: str, title: str, source_file: str, source_url: str, tokenizer) -> list[dict]:
    original_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = max(original_max_length, 1_000_000)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    tokenizer.model_max_length = original_max_length
    offsets = encoded["offset_mapping"]
    token_count = len(offsets)
    if not token_count:
        return []
    token_ends = [end for _, end in offsets]
    chapter_match = re.match(r"^(\d+)\s+", title)
    chapter_number = int(chapter_match.group(1)) if chapter_match else None
    chunks = []
    start = 0

    while start < token_count:
        body_start = offsets[start][0]
        section = section_at(text, body_start, title)
        section_path = title if section == title else f"{title} > {section}"
        context = f"Chapter: {title}\nSection: {section}\n\n"
        context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
        body_limit = max(1, CHUNK_TOKENS - context_tokens - 8)
        target_end = min(token_count, start + body_limit)

        if target_end < token_count:
            target_char_end = offsets[target_end - 1][1]
            min_end = max(start + 1, target_end - min(100, body_limit // 3))
            adjusted_end = sentence_boundary(
                token_ends, text, body_start, target_char_end, min_end, target_end
            )
            if adjusted_end is not None and adjusted_end > start:
                target_end = adjusted_end

        body_end = offsets[target_end - 1][1]
        body = text[body_start:body_end].strip()
        chunk_text = context + body
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        point_id = str(uuid.uuid5(NAMESPACE, f"{source_file}:{len(chunks)}:{content_hash}"))
        chunks.append({
            "id": point_id,
            "text": chunk_text,
            "source_title": "Mineralogy (OpenGeology)",
            "chapter_number": chapter_number,
            "chapter_title": title,
            "section_path": section_path,
            "source_file": source_file,
            "source_url": source_url,
            "chunk_index": len(chunks),
            "content_hash": content_hash,
            "embedding_model": MODEL_NAME,
            "embedding_version": "BAAI/bge-m3@main",
            "embedding_dimension": VECTOR_SIZE,
            "chunking_version": CHUNKING_VERSION,
            "content_type": "text",
        })

        if target_end >= token_count:
            break
        start = max(start + 1, target_end - OVERLAP_TOKENS)

    return chunks


def ensure_collection() -> None:
    existing = qdrant_request("GET", "/collections")["result"]["collections"]
    names = {item["name"] for item in existing}
    if COLLECTION not in names:
        qdrant_request("PUT", f"/collections/{COLLECTION}", {
            "vectors": {"size": VECTOR_SIZE, "distance": "Cosine"},
        })
        print(f"Created collection {COLLECTION} ({VECTOR_SIZE}d cosine)")
    info = qdrant_request("GET", f"/collections/{COLLECTION}")["result"]
    vector_config = info["config"]["params"]["vectors"]
    if vector_config.get("size") != VECTOR_SIZE or vector_config.get("distance") != "Cosine":
        raise RuntimeError(f"Existing collection {COLLECTION} has an incompatible vector configuration.")
    if COLLECTION in names:
        print(f"Using existing collection {COLLECTION}")

    for field_name, field_schema in (("chapter_number", "integer"), ("source_file", "keyword")):
        if field_name not in info.get("payload_schema", {}):
            qdrant_request("PUT", f"/collections/{COLLECTION}/index", {
                "field_name": field_name,
                "field_schema": field_schema,
            })


def embedding_revision() -> str:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        ref = Path(HF_HUB_CACHE) / "models--BAAI--bge-m3" / "refs" / "main"
        if ref.is_file():
            return f"{MODEL_NAME}@{ref.read_text(encoding='utf-8').strip()}"
    except (ImportError, OSError):
        pass
    return f"{MODEL_NAME}@main"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="count chunks without writing to Qdrant")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_text_file = {page["text_file"]: page for page in manifest["pages"] if page.get("text_file")}
    sources = sorted(SAVED_PAGES.glob("*.txt"))
    if not sources:
        raise RuntimeError(f"No .txt files found in {SAVED_PAGES}")

    model = None
    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    else:
        model = load_model("auto")
        tokenizer = model.tokenizer
    version = embedding_revision()
    all_chunks = []
    for path in sources:
        source_info = by_text_file.get(path.name, {})
        title = source_info.get("title", path.stem)
        source_file = str(path.relative_to(ROOT))
        chunks = chunk_document(
            path.read_text(encoding="utf-8"),
            title,
            source_file,
            source_info.get("url", ""),
            tokenizer,
        )
        for chunk in chunks:
            chunk["embedding_version"] = version
        all_chunks.extend(chunks)
        print(f"CHUNKED {title}: {len(chunks)} chunks")

    lengths = [len(tokenizer.encode(item["text"], add_special_tokens=False)) for item in all_chunks]
    if lengths:
        print(f"Chunk token lengths: min={min(lengths)} max={max(lengths)} avg={sum(lengths) / len(lengths):.1f}")
    print(f"Prepared {len(all_chunks)} chunks from {len(sources)} text files")
    if args.dry_run:
        return

    ensure_collection()
    for offset in range(0, len(all_chunks), args.batch_size):
        batch = all_chunks[offset:offset + args.batch_size]
        vectors = embed_texts(model, [item["text"] for item in batch])
        points = []
        for item, vector in zip(batch, vectors, strict=True):
            points.append({
                "id": item["id"],
                "vector": vector.tolist(),
                "payload": {key: value for key, value in item.items() if key != "id"},
            })
        qdrant_request("PUT", f"/collections/{COLLECTION}/points?wait=true", {"points": points})
        done = min(offset + len(batch), len(all_chunks))
        print(f"UPSERTED {done}/{len(all_chunks)}")

    info = qdrant_request("GET", f"/collections/{COLLECTION}")["result"]
    print(f"DONE collection={COLLECTION} points={info['points_count']} status={info['status']}")


if __name__ == "__main__":
    main()
