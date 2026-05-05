# scripts/crawl_data.py
"""Utility to crawl a Wikipedia page (or any MediaWiki site) and save the article
content as a Markdown file in the local ``data`` directory.

Usage::

    python -m scripts.crawl_data <wiki_url> [--out-dir data]

The script fetches the HTML of the given page, extracts the main article text,
converts it to markdown using ``markdownify`` and writes the result to a file
named after the page title (sanitized).  The output directory is created if it
does not exist.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
from bs4 import BeautifulSoup

# Optional: ``markdownify`` provides HTML‑to‑Markdown conversion.  It is a
# lightweight dependency; if unavailable we fall back to plain text extraction.
try:
    from markdownify import markdownify as mdify
except ImportError:  # pragma: no cover
    mdify = None


def sanitize_filename(name: str) -> str:
    """Return a filesystem‑safe filename derived from *name*.

    Non‑alphanumeric characters are replaced with ``_`` and the result is
    stripped of leading/trailing underscores.
    """
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    return safe.strip("_")


def fetch_html(url: str) -> str:
    """Download the HTML content at *url*.

    Raises ``requests.HTTPError`` on network problems or non‑200 responses.
    """
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.text


def extract_article(html: str) -> tuple[str, str]:
    """Extract the article title and body from MediaWiki HTML.

    Returns ``(title, markdown)`` where *markdown* is the converted article
    content.  If ``markdownify`` is unavailable the plain text is returned.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Title – typically in the <h1 id="firstHeading">
    title_tag = soup.find("h1", {"id": "firstHeading"})
    title = title_tag.get_text(strip=True) if title_tag else "unknown"
    # Main content – in <div class="mw-parser-output">
    content_div = soup.find("div", {"class": "mw-parser-output"})
    if not content_div:
        # Fallback to the whole body
        content_div = soup.body
    html_body = str(content_div)
    if mdify:
        markdown = mdify(html_body, heading_style="ATX")
    else:
        # Simple fallback: strip tags and keep line breaks
        markdown = content_div.get_text(separator="\n", strip=True)
    return title, markdown


def save_markdown(title: str, markdown: str, out_dir: Path) -> Path:
    """Write *markdown* to ``out_dir/<sanitized_title>.md`` and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = sanitize_filename(title) or "article"
    out_path = out_dir / f"{filename}.md"
    out_path.write_text(markdown, encoding="utf-8")
    return out_path


def crawl_wiki(url: str, out_dir: Path) -> Path:
    """Fetch *url*, extract the article and store it as markdown.

    Returns the path to the created file.
    """
    html = fetch_html(url)
    title, markdown = extract_article(html)
    return save_markdown(title, markdown, out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Crawl a MediaWiki page and save as markdown.")
    parser.add_argument("url", help="Full URL of the wiki page to crawl.")
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Directory to store the generated .md files (default: data).",
    )
    args = parser.parse_args(argv)
    out_path = crawl_wiki(args.url, Path(args.out_dir))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
