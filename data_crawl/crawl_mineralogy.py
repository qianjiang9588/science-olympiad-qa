#!/usr/bin/env python3
"""Download OpenGeology Mineralogy pages, extracted text, and referenced images."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

START_URL = "https://opengeology.org/Mineralogy/"
SITE_HOST = "opengeology.org"
SECTION_PATH = "/Mineralogy/"
USER_AGENT = "OpenGeologyMineralogyResearchCrawler/1.0 (polite educational content archiver)"


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in {"fbclid", "gclid"}])
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def in_scope(url: str) -> bool:
    p = urlsplit(url)
    return p.scheme in {"http", "https"} and p.netloc.lower().removeprefix("www.") == SITE_HOST and p.path.startswith(SECTION_PATH)


def slug_for(url: str) -> str:
    path = urlsplit(url).path.strip("/")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", path.replace("/", "-")).strip("-") or "home"
    return f"{slug[:100]}-{hashlib.sha256(url.encode()).hexdigest()[:10]}"


def text_of(node) -> str:
    return re.sub(r"\n{3,}", "\n\n", node.get_text("\n", strip=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/opengeology_mineralogy"))
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means no page limit")
    args = parser.parse_args()

    root = args.output
    html_dir, image_dir, docs_dir = root / "raw/html", root / "raw/images", root / "processed/pages"
    for directory in (html_dir, image_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,image/*;q=0.8,*/*;q=0.5"})
    queue = deque([canonical_url(START_URL)])
    seen: set[str] = set()
    records = []

    while queue and (not args.max_pages or len(seen) < args.max_pages):
        url = queue.popleft()
        if url in seen or not in_scope(url):
            continue
        seen.add(url)
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"PAGE ERROR {url}: {exc}")
            continue
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type:
            continue
        final_url = canonical_url(response.url)
        if not in_scope(final_url):
            continue
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else final_url
        if "making sure you're not a bot" in title.lower() or "proof of work" in response.text.lower():
            print(f"BLOCKED {final_url}: site proof-of-work check returned instead of the page")
            continue
        main = soup.find("main") or soup.find("article") or soup.find(id=re.compile(r"content|main", re.I)) or soup.body or soup
        for el in main.select("script, style, noscript, nav, footer, header, .sharedaddy, .jp-relatedposts"):
            el.decompose()
        # Keep page heading hierarchy and figure text in extracted text.
        page_text = text_of(main)
        key = slug_for(final_url)
        html_path = html_dir / f"{key}.html"
        html_path.write_bytes(response.content)
        image_records = []

        for img in main.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if not src:
                srcset = img.get("srcset") or img.get("data-srcset") or ""
                candidates = [x.strip().split() for x in srcset.split(",") if x.strip()]
                src = candidates[-1][0] if candidates else None
            if not src:
                continue
            image_url = urljoin(final_url, src)
            if urlsplit(image_url).scheme not in {"http", "https"}:
                continue
            figure = img.find_parent("figure")
            caption = text_of(figure.find("figcaption")) if figure and figure.find("figcaption") else ""
            if not caption and figure:
                caption = text_of(figure)
            alt = img.get("alt", "").strip()
            try:
                ir = session.get(image_url, timeout=30)
                ir.raise_for_status()
                mime = ir.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if not mime.startswith("image/"):
                    continue
                ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp", "image/svg+xml": ".svg", "image/tiff": ".tif"}.get(mime, Path(urlsplit(image_url).path).suffix.lower() or ".img")
                image_name = hashlib.sha256(image_url.encode()).hexdigest()[:20] + ext
                local_image = image_dir / image_name
                if not local_image.exists():
                    local_image.write_bytes(ir.content)
                image_records.append({"url": image_url, "local_path": str(local_image.relative_to(root)), "alt": alt, "caption": caption, "mime_type": mime})
            except requests.RequestException as exc:
                print(f"IMAGE ERROR {image_url}: {exc}")
            time.sleep(args.delay)

        record = {
            "url": final_url,
            "title": title,
            "text": page_text,
            "html_path": str(html_path.relative_to(root)),
            "images": image_records,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
        }
        (docs_dir / f"{key}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
        print(f"PAGE {len(records)} | {title} | {len(image_records)} images")

        # Discover only canonical internal pages under the requested section.
        for link in main.find_all("a", href=True):
            target = canonical_url(urljoin(final_url, link["href"]))
            if in_scope(target) and target not in seen:
                queue.append(target)
        time.sleep(args.delay)

    manifest = {
        "source": START_URL,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(records),
        "image_count": sum(len(r["images"]) for r in records),
        "pages": [{"url": r["url"], "title": r["title"], "json_path": str((docs_dir / f"{slug_for(r['url'])}.json").relative_to(root))} for r in records],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Convenient combined export for downstream chunking/indexing.
    (root / "processed/documents.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE pages={manifest['page_count']} images={manifest['image_count']} output={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
