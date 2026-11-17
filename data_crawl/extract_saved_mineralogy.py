#!/usr/bin/env python3
"""Extract readable text from locally saved OpenGeology Mineralogy HTML pages."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent


def extract_text(path: Path) -> dict[str, str]:
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else path.stem
    source = soup.find("meta", property="og:url") or soup.find("link", rel="canonical")
    url = source.get("content") or source.get("href") if source else ""

    content = soup.find("article") or soup.find("main") or soup.find(id="content") or soup.body or soup
    for node in content.select(
        "script, style, noscript, nav, footer, header, aside, .sharedaddy, "
        ".jp-relatedposts, .entry-footer, .post-navigation, .comments-area"
    ):
        node.decompose()
    text = content.get_text("\n", strip=True)
    text = re.sub(r"[\t\xa0 ]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return {"title": title, "url": url, "text": text}


def main() -> None:
    source_dir = ROOT / "data" / "downloaded"
    output_dir = ROOT / "data" / "opengeology_mineralogy" / "processed" / "saved_pages"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=source_dir, help="directory containing saved HTML pages")
    parser.add_argument("--output", type=Path, default=output_dir, help="directory for extracted text")
    args = parser.parse_args()

    source_dir = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(source_dir.glob("*.html"))
    pages = []
    for source in sources:
        page = extract_text(source)
        page["source_file"] = str(source)
        page["text_file"] = source.stem + ".txt"
        (output_dir / page["text_file"]).write_text(page["text"] + "\n", encoding="utf-8")
        pages.append(page)

    combined = "\n\n\n".join(f"# {p['title']}\n\n{p['text']}" for p in pages)
    combined_path = output_dir / "all_pages.md"
    combined_path.write_text(combined + "\n", encoding="utf-8")
    manifest = {
        "source_directory": str(source_dir),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "character_count": sum(len(p["text"]) for p in pages),
        "pages": pages,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {len(pages)} pages, {manifest['character_count']:,} text characters")
    print(f"Combined text: {combined_path}")


if __name__ == "__main__":
    main()
