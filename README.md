# science-olympiad-qa
A chatbot to help student study different topics for science-olympiad competition

## OpenGeology Mineralogy source archive

`data_crawl/crawl_mineralogy.py` crawls pages below `https://opengeology.org/Mineralogy/`, saving original HTML, extracted page text, image files, and image/page metadata under `data/opengeology_mineralogy/`.

Install dependencies with `python3 -m pip install -r requirements.txt`, then run `python3 data_crawl/crawl_mineralogy.py`. Use `--max-pages N` for a limited crawl or `--output PATH` to choose another destination. The crawler stays within the Mineralogy path, uses a descriptive User-Agent, and pauses between requests. If the site returns its browser proof-of-work page, the crawler reports that and skips it.

If you saved the site pages with your browser, place the HTML files in `data/downloaded/` and run `python3 data_crawl/extract_saved_mineralogy.py`. The script writes per-page text, a combined `all_pages.md`, and a manifest under `data/opengeology_mineralogy/processed/saved_pages/`. Use `--input PATH` or `--output PATH` to select different directories.

## Local BGE-M3 embeddings

Create an isolated environment and install the optional embedding dependency:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-local-embeddings.txt
```

Run local inference on text (the first run downloads the complete Hugging Face model snapshot, including weights, tokenizer, and configuration files, into the Hugging Face cache):

```sh
.venv/bin/python data_crawl/local_embeddings.py --text "What is a polymorph?"
```

Use `--file PATH` to embed one UTF-8 text file, `--output PATH` to save the resulting JSON, or `--device cpu` to force CPU inference. The script selects Apple Silicon MPS automatically when PyTorch reports it is available. The output contains a normalized 1,024-dimensional dense vector, ready for Qdrant cosine search. Load the model once and reuse it for batches when embedding a corpus.

For a browser interface, start the local service:

```sh
.venv/bin/python data_crawl/local_embedding_server.py
```

Open `http://127.0.0.1:8765/`, enter text, and click **Generate embedding**. The service loads the model once, then runs inference locally. Stop it with `Ctrl+C` in the terminal.

To index the saved pages into local Qdrant, first preview the chunk count with `HF_HUB_OFFLINE=1 .venv/bin/python data_crawl/index_saved_pages.py --dry-run`, then run `HF_HUB_OFFLINE=1 .venv/bin/python data_crawl/index_saved_pages.py`. This creates or reuses `mineralogy_chunks_v1`, uses 500-token section-aware chunks with approximately 70-token overlap, and upserts normalized 1,024-dimensional BGE-M3 vectors with source metadata.

### Inspect the Qdrant collection

With the local Qdrant service running at `127.0.0.1:6333`, inspect collection status, vector configuration, and point count:

```sh
curl -fsS http://127.0.0.1:6333/collections/mineralogy_chunks_v1 | python3 -m json.tool
```

To view a few stored chunks and their source metadata without returning the vector values:

```sh
curl -fsS -X POST http://127.0.0.1:6333/collections/mineralogy_chunks_v1/points/scroll \
  -H 'Content-Type: application/json' \
  -d '{"limit":3,"with_payload":["chapter_title","section_path","source_file","source_url","text"],"with_vector":false}' \
  | python3 -m json.tool
```

The scroll endpoint returns points in pages. Use the response's `next_page_offset` as the `offset` in the next request to continue through the collection. The current collection has 1,070 points; the count changes if the saved pages are re-indexed.

Qdrant normally provides a Web UI at `http://127.0.0.1:6333/dashboard`. The standalone binary used for this local setup does not include the dashboard's static files, so use the REST commands above to inspect this instance.
