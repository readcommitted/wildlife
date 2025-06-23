"""
species_embeddings.py — Species Scraping & Embedding Automation
---------------------------------------------------------------

Automates the process of:
✅ Scraping species names from Wikipedia list or individual pages
✅ Downloading representative images via Wikipedia API
✅ Generating 512-dim CLIP embeddings for species images
✅ Storing results in `wildlife.species_embedding` with metadata
✅ Supports manual or UI-driven processing for pending records

This forms the species reference library for fast visual similarity searches.

Requirements:
- Wikipedia API access for images
- Local CLIP model via `generate_image_embedding()`
- SQLAlchemy ORM for database interaction
- Streamlit UI for progress feedback

Author: Matt Scardino
Project: Wildlife Vision System
"""

import os
from pathlib import Path
from io import BytesIO
from bs4 import BeautifulSoup
import streamlit as st
from PIL import Image
import requests
from db.db import SessionLocal
from db.species_model import SpeciesEmbedding
from tools.embedding_utils import generate_image_embedding
from db.scrape_model import ScrapeSource
from config.settings import MEDIA_ROOT, IMAGE_DIR, WIKI_API_URL, HEADERS


os.makedirs(IMAGE_DIR, exist_ok=True)


def get_active_sources():
    """
    Fetch all active ScrapeSource records (e.g., Wikipedia regions to scrape).
    """
    session = SessionLocal()
    try:
        return session.query(ScrapeSource).filter_by(active=True).all()
    finally:
        session.close()


def scrape_species(url: str) -> set[str]:
    """
    Scrape species names from a Wikipedia list page.

    Args:
        url (str): Wikipedia list page URL

    Returns:
        set[str]: Unique species names extracted from the list
    """
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    species = set()

    content = soup.find("div", id="mw-content-text")
    if content:
        for ul in content.find_all("ul", recursive=False):
            for li in ul.find_all("li", recursive=False):
                name = li.get_text().split("(")[0].strip()
                if name:
                    species.add(name)

    for ul in soup.select("div.div-col ul"):
        for li in ul.find_all("li"):
            name = li.get_text().split("(")[0].strip()
            if name:
                species.add(name)

    return species


def extract_species_name_from_species_page(url: str) -> str | None:
    """
    Extract species common and scientific name from a Wikipedia species page.

    Args:
        url (str): Wikipedia species page URL

    Returns:
        str | None: Combined name string, or None if extraction fails
    """
    resp = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")

    common_name = soup.find("h1", id="firstHeading").text.strip()
    sci = soup.select_one(".infobox i, .infobox span[class*='Binomial']")
    sci_name = sci.text.strip() if sci else ""

    status_cell = soup.select_one(".infobox td:contains('Least Concern'), .infobox td:contains('LC')")
    status = "LC" if status_cell else ""

    return f"{common_name}, {sci_name} {status}".strip()


def fetch_species_image(name: str, category: str, api_url: str) -> str | None:
    """
    Fetch and save species image from Wikipedia API.

    Args:
        name (str): Species name
        category (str): Taxonomic category (birds, mammals, etc.)
        api_url (str): Wikipedia API endpoint

    Returns:
        str | None: Saved image path, or None if failed
    """
    title = name.split(",")[0].strip()
    slug = title.lower().replace(" ", "_")
    save_dir = IMAGE_DIR / category
    save_path = save_dir / f"{slug}.jpg"
    os.makedirs(save_dir, exist_ok=True)

    if save_path.exists():
        return str(save_path)

    try:
        resp = requests.get(api_url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            img_url = page.get("original", {}).get("source") or page.get("thumbnail", {}).get("source")
            if not img_url:
                continue

            img_resp = requests.get(img_url, headers=HEADERS)
            img_resp.raise_for_status()

            if "image" not in img_resp.headers.get("Content-Type", "").lower():
                return None

            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            img.save(save_path)
            return str(save_path)

    except Exception as e:
        print(f"Image fetch failed for {name}: {e}")

    return None


def process_pending():
    """
    Processes all species records missing embeddings:
    - Downloads image
    - Generates embedding
    - Updates DB with image path and vector
    """
    session = SessionLocal()
    try:
        pending = session.query(SpeciesEmbedding).filter(SpeciesEmbedding.image_embedding.is_(None)).all()
    except Exception as e:
        session.close()
        print(f"ERROR: Could not load pending records: {e}")
        return

    prog = st.progress(0)
    log = st.empty()
    success = fail = 0

    for idx, rec in enumerate(pending, start=1):
        img_path = fetch_species_image(rec.species, rec.category, rec.api_url)
        session_up = SessionLocal()

        try:
            rec_up = session_up.get(SpeciesEmbedding, rec.id)
            if not img_path:
                rec_up.status = 'no_image'
                session_up.commit()
                fail += 1
                continue

            vec = generate_image_embedding(img_path)
            rec_up.image_path = img_path
            rec_up.image_embedding = vec.tolist()
            rec_up.status = 'embedded'
            session_up.commit()
            success += 1

        except Exception as e:
            session_up.rollback()
            rec_up = session_up.get(SpeciesEmbedding, rec.id)
            rec_up.status = 'error'
            session_up.commit()
            print(f"ERROR: Embedding failed for {rec.species}: {e}")
            fail += 1

        finally:
            session_up.close()

        prog.progress(idx / len(pending))
        log.text(f"{idx}/{len(pending)} ✔️{success} ❌{fail}")

    session.close()
    st.success(f"✅ Completed: {success} succeeded, {fail} failed.")


def run_pipeline():
    """
    Streamlit UI to run pending species embedding process.
    """
    st.title("🦌 Species Embeddings Automation")
    if st.button("Process pending embeddings"):
        process_pending()


if __name__ == "__main__":
    run_pipeline()
