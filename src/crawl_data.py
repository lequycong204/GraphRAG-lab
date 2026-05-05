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
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

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


def get_wiki_title(url: str) -> str:
    parsed_url = urlparse(url)
    return unquote(parsed_url.path.removeprefix("/wiki/")).replace("_", " ")


def get_api_url(url: str) -> str:
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}/w/api.php"


def fetch_page_from_api(url: str) -> tuple[str, str]:
    """Fetch rendered page HTML from the MediaWiki API."""
    headers = {
        "User-Agent": "GraphRAG-lab/1.0 (local educational crawler)",
        "Accept": "application/json",
    }
    params = {
        "action": "parse",
        "page": get_wiki_title(url),
        "prop": "text|displaytitle",
        "format": "json",
        "redirects": "1",
    }
    resp = requests.get(get_api_url(url), params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("info", "MediaWiki API error"))
    title = BeautifulSoup(data["parse"]["displaytitle"], "html.parser").get_text(strip=True)
    html = data["parse"]["text"]["*"]
    return title, html


def html_to_text(html: str) -> str:
    """Convert rendered MediaWiki HTML to plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select(
        "script, style, img, figure, table, sup, .mw-editsection, .reference, .reflist, .navbox, .metadata"
    ):
        tag.decompose()
    for tag in soup.find_all("a"):
        tag.replace_with(tag.get_text(" ", strip=True))
    block_tags = soup.select("h1, h2, h3, h4, h5, h6, p, li")
    lines = [tag.get_text(" ", strip=True) for tag in block_tags]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\[(?:\d+|cần dẫn nguồn)\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def save_markdown(title: str, markdown: str, out_dir: Path) -> Path:
    """Write *markdown* to ``out_dir/<sanitized_title>.md`` and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = sanitize_filename(title) or "article"
    out_path = out_dir / f"{filename}.md"
    out_path.write_text(markdown, encoding="utf-8")
    return out_path


def filter_vietnamese(text: str) -> str:
    """Keep readable content lines from Vietnamese pages."""
    content_regex = re.compile(r"[A-Za-zÀ-ỹ]")
    junk_regex = re.compile(r"^(\^|\[|\]|\||•|↑|Tham khảo|Liên kết ngoài|Xem thêm)\s*$")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or junk_regex.match(line):
            continue
        if content_regex.search(line):
            lines.append(line)
    return "\n".join(lines)


def crawl_wiki(url: str, out_dir: Path) -> Path:
    """Fetch *url*, extract the article and store it as markdown.

    Returns the path to the created file.
    """
    title, html = fetch_page_from_api(url)
    text = html_to_text(html)
    text = filter_vietnamese(text)
    if out_dir is None:
        out_dir = Path("data")
    return save_markdown(title, text, out_dir)


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
