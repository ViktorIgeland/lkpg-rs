"""
Scrape Linköping Municipality news and index into Pinecone with OpenAI embeddings.

Steps:
- Scrape main page for news entries (title, date, url)
- Visit each detail page to extract content/body
- Clean text and normalize dates
- Embed (title + content) with text-embedding-3-small
- Upsert into Pinecone with metadata (title, date, url)
"""

import json
import os
import time
import hashlib
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from .utils import clean_text, normalize_date


BASE_URL = "https://www.linkoping.se/nyheter/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}


def absolute_url(href: str) -> str:
    if href.startswith("http"):
        return href
    return f"https://www.linkoping.se{href}"


def extract_main_items(soup: BeautifulSoup, max_items: int = 5) -> List[Dict[str, Optional[str]]]:
    """Heuristic extraction from the main news listing page."""
    items: List[Dict[str, Optional[str]]] = []
    seen_urls = set()
    # Find links that look like news detail pages
    for a in soup.select('a[href^="/nyheter/"], a[href^="https://www.linkoping.se/nyheter/"]'):
        href = a.get("href")
        if not href:
            continue
        url = absolute_url(href)
        if url in seen_urls:
            continue

        # Try to find a reasonable container for title/date (article/li/div)
        container = a
        for _ in range(3):
            if container.parent is None:
                break
            container = container.parent
            if container.name in {"article", "li", "div"}:
                break

        # Title heuristic: prefer headings then link text
        title_el = container.find(["h1", "h2", "h3"]) if container else None
        title_text = clean_text(title_el.get_text(" ")) if title_el else clean_text(a.get_text(" "))
        title_text = title_text.strip()
        if not title_text:
            continue

        # Date heuristic: prefer <time> datetime or text near the link
        date_text = None
        time_el = container.find("time") if container else None
        if time_el:
            date_text = time_el.get("datetime") or time_el.get_text(" ")
        if not date_text:
            # Look for date pattern close to link
            sibling_text = clean_text(container.get_text(" ")) if container else ""
            date_text = sibling_text

        items.append({
            "title": title_text,
            "date_raw": date_text,
            "url": url,
        })
        seen_urls.add(url)
        if len(items) >= max_items:
            break
    return items


def extract_detail_content(detail_html: str) -> str:
    """Extract main body content from a detail page."""
    soup = BeautifulSoup(detail_html, "html.parser")

    # Prefer article content
    article = soup.find("article")
    if article:
        paragraphs = [p.get_text(" ") for p in article.find_all("p")]
        text = " ".join(paragraphs) or article.get_text(" ")
        return clean_text(text)

    # Fallback to main
    main = soup.find("main")
    if main:
        paragraphs = [p.get_text(" ") for p in main.find_all("p")]
        text = " ".join(paragraphs) or main.get_text(" ")
        return clean_text(text)

    # As a last resort, take all paragraphs
    paragraphs = [p.get_text(" ") for p in soup.find_all("p")]
    text = " ".join(paragraphs) or soup.get_text(" ")
    return clean_text(text)


def scrape() -> List[Dict[str, str]]:
    """Scrape the listing page and all detail pages."""
    resp = requests.get(BASE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    main_items = extract_main_items(soup)
    results: List[Dict[str, str]] = []

    for item in main_items:
        url = item["url"]
        try:
            d_resp = requests.get(url, headers=HEADERS, timeout=30)
            d_resp.raise_for_status()
            content = extract_detail_content(d_resp.text)
        except Exception as e:
            print(f"Failed to fetch detail page {url}: {e}")
            content = ""

        iso_date = normalize_date(item.get("date_raw") or "") or ""
        result = {
            "id": hashlib.md5(url.encode("utf-8")).hexdigest(),
            "title": item.get("title") or "",
            "date": iso_date,
            "url": url,
            "content": content,
        }
        results.append(result)
        # Be polite to the server
        time.sleep(0.5)

    return results


def ensure_pinecone_index(pc: Pinecone, index_name: str, dimension: int, cloud: str, region: str) -> None:
    li = pc.list_indexes()
    # Support multiple SDK return shapes
    names = set()
    try:
        # Newer SDKs may have .names()
        if hasattr(li, "names"):
            names = set(li.names())  # type: ignore[attr-defined]
        else:
            names = {getattr(i, "name", i.get("name")) for i in li}  # type: ignore[union-attr]
    except Exception:
        # Fallback: attempt iteration as strings
        try:
            names = set(li)  # type: ignore[arg-type]
        except Exception:
            names = set()

    if index_name not in names:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    # Wait for readiness
    while True:
        desc = pc.describe_index(index_name)
        ready = False
        try:
            status = getattr(desc, "status", None)
            if isinstance(status, dict):
                ready = bool(status.get("ready"))
            else:
                ready = bool(getattr(status, "ready", False))
        except Exception:
            ready = False
        if ready:
            break
        time.sleep(1)


def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    # For simplicity we embed sequentially to keep the code minimal
    embeddings: List[List[float]] = []
    for t in texts:
        resp = client.embeddings.create(model="text-embedding-3-small", input=t)
        embeddings.append(resp.data[0].embedding)
    return embeddings


def upsert_to_pinecone(pc: Pinecone, index_name: str, articles: List[Dict[str, str]], client: OpenAI) -> None:
    index = pc.Index(index_name)
    inputs = [f"{a['title']}\n\n{a['content']}" for a in articles]
    vectors = embed_texts(client, inputs)
    upserts = []
    for article, vector in zip(articles, vectors):
        upserts.append({
            "id": article["id"],
            "values": vector,
            "metadata": {
                "title": article["title"],
                "date": article["date"],
                "url": article["url"],
            },
        })
    # Upsert in a single call (small dataset). For larger sets, batch.
    index.upsert(vectors=upserts)


def save_json(path: str, data: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    load_dotenv()

    print("Scraping news listing and detail pages...")
    articles = scrape()
    print(f"Scraped {len(articles)} articles")

    save_json("data/news.json", articles)
    print("Saved raw articles to data/news.json")

    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY must be set")

    client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)

    index_name = os.getenv("PINECONE_INDEX_NAME", "linkoping")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "eu-west-1")

    # text-embedding-3-small has 1536 dimensions
    ensure_pinecone_index(pc, index_name=index_name, dimension=1536, cloud=cloud, region=region)
    print(f"Index '{index_name}' is ready")

    print("Creating embeddings and upserting to Pinecone...")
    upsert_to_pinecone(pc, index_name, articles, client)
    print("Upsert completed")


if __name__ == "__main__":
    main()


