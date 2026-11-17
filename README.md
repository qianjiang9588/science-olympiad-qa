# science-olympiad-qa
A chatbot to help student study different topics for science-olympiad competition

## OpenGeology Mineralogy source archive

`data_crawl/crawl_mineralogy.py` crawls pages below `https://opengeology.org/Mineralogy/`, saving original HTML, extracted page text, image files, and image/page metadata under `data/opengeology_mineralogy/`.

Install dependencies with `python3 -m pip install -r requirements.txt`, then run `python3 data_crawl/crawl_mineralogy.py`. Use `--max-pages N` for a limited crawl or `--output PATH` to choose another destination. The crawler stays within the Mineralogy path, uses a descriptive User-Agent, and pauses between requests. If the site returns its browser proof-of-work page, the crawler reports that and skips it.

If you saved the site pages with your browser, place the HTML files in `data/downloaded/` and run `python3 data_crawl/extract_saved_mineralogy.py`. The script writes per-page text, a combined `all_pages.md`, and a manifest under `data/opengeology_mineralogy/processed/saved_pages/`. Use `--input PATH` or `--output PATH` to select different directories.
