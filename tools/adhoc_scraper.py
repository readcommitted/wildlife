"""
adhoc_scraper.py — Ad-Hoc Wikipedia Species Scraper & Embedder
---------------------------------------------------------------

This Streamlit UI allows users to scrape species names and images from Wikipedia,
generate CLIP image embeddings, and insert records into the species_embedding table.

Supports:
- Scraping from individual species pages or Wikipedia list pages
- Fetching representative species images via the Wikipedia API
- Generating CLIP embeddings using the project's ingest pipeline
- Automatically updating species metadata via stored procedures

Intended for:
- Rapid addition of new species to the Wildlife Vision System
- Filling missing embeddings for species already present

Dependencies:
- Streamlit UI
- SQLAlchemy ORM
- Wikipedia API (read-only)
- Project image embedding tools

Author: Matt Scardino
Project: Wildlife Vision System
"""

import streamlit as st
from db.db import SessionLocal
from db.species_model import SpeciesEmbedding
from tools.species_embeddings import (
    fetch_species_image,
    scrape_species,
    extract_species_name_from_species_page,
)
import requests
from sqlalchemy import text
from pathlib import Path
from core.ingest import generate_image_embedding
from config.settings import MEDIA_ROOT, WIKI_API_URL, HEADERS, IMAGE_DIR


# --- Streamlit UI ---
url = st.text_input("Wikipedia species or list URL", placeholder="https://en.wikipedia.org/wiki/Great_grey_owl")
category = st.selectbox("Category", ["birds", "mammals"])

# --- Scrape and Process Handler ---
if st.button("Scrape and Process"):
    if not url:
        st.warning("Please enter a valid URL.")
    else:
        session = SessionLocal()
        species_set = set()

        if "list_of_" in url.lower() or "list" in url.lower():
            st.info("Detected Wikipedia list page")
            species_set = scrape_species(url)
        else:
            st.info("Detected individual species page")
            single_name = extract_species_name_from_species_page(url)
            species_set = [single_name] if single_name else []

        total = len(species_set)
        prog = st.progress(0)
        log = st.empty()
        success = fail = skip = 0

        for idx, name in enumerate(sorted(species_set), start=1):
            st.write(f"Processing species {idx}/{total}: {name}")

            # Skip if species already embedded
            rec = session.query(SpeciesEmbedding).filter_by(common_name=name).first()
            if rec and rec.image_embedding:
                skip += 1
                continue

            title = name.split(",")[0].strip()
            params = {
                "action": "query",
                "titles": title,
                "prop": "pageimages",
                "format": "json",
                "piprop": "original|thumbnail",
                "pithumbsize": 500
            }
            api_url = requests.Request("GET", WIKI_API_URL, params=params).prepare().url

            if not rec:
                rec = SpeciesEmbedding(common_name=name, category=category, api_url=api_url)
                session.add(rec)
                session.commit()

            img_path = fetch_species_image(name, category, api_url)
            if not img_path:
                rec.status = 'no_image'
                session.commit()
                fail += 1
                continue

            try:
                vec = generate_image_embedding(img_path)
                rec.image_path = img_path
                rec.image_embedding = vec
                rec.status = 'embedded'
                session.commit()
                success += 1
            except Exception as e:
                rec.status = 'error'
                session.commit()
                fail += 1

            prog.progress(idx / total)
            log.text(f"{idx}/{total} ✔️{success} ❌{fail} ⏭️{skip}")

        # --- Final Metadata Sync ---
        try:
            st.info("Running stored procedure `update_species_fields()`...")
            session.execute(text("SELECT update_species_fields();"))
            session.commit()
            st.success("Species fields successfully updated.")
        except Exception as e:
            session.rollback()
            st.error(f"Error occurred: {e}")

        session.close()
        st.success(f"Done. Success: {success}, Failures: {fail}, Skipped: {skip}")
