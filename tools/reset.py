"""
reset.py — Destructive Data Reset Utility
------------------------------------------

⚠️ USE WITH EXTREME CAUTION ⚠️

This Streamlit module provides dangerous but useful options for developers during testing:

✅ Bulk deletes all image-related records (non-destructive to schema)
✅ Drops all image-related tables entirely
✅ Deletes SpeciesNet results JSON file
✅ Clears raw and processed data lake directories

Primarily intended for development environments only.

Requirements:
- SQLAlchemy for database access
- pathlib and shutil for file handling
- Streamlit UI

Author: Matt Scardino
Project: Wildlife Vision System
"""

import shutil
import streamlit as st
from pathlib import Path
from db.db import SessionLocal
import os
from sqlalchemy import text
from config.settings import MEDIA_ROOT, STAGE_PROCESSED_DIR, PREDICTIONS_JSON, RAW_DIR, JPG_DIR


# --- Initialize Session ---
session = SessionLocal()

st.write("⚠️ Use caution — destructive operations ahead.")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

# --- Button Controls ---
with col1:
    delete_clicked = st.button("🔥 Delete All Image Data")
with col2:
    drop_clicked = st.button("💣 Drop All Image Tables")
with col3:
    species_clicked = st.button("🗑️ Delete SpeciesNet JSON")
with col4:
    datalake_clicked = st.button("🗂️ Clear Data Lake Paths")


# --- Delete All Image Records (Preserve Schema) ---
if delete_clicked:
    tables = [
        "wildlife.image_label",
        "wildlife.image_embedding",
        "wildlife.image_exif",
        "wildlife.image_header"
    ]
    for table in tables:
        session.execute(text(f"DELETE FROM {table}"))
    session.commit()
    st.success("✅ All image-related records deleted successfully.")


# --- Drop Image Tables (Remove Schema) ---
if drop_clicked:
    tables = [
        "wildlife.image_label",
        "wildlife.image_embedding",
        "wildlife.image_exif",
        "wildlife.image_header"
    ]
    for table in tables:
        session.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    session.commit()
    st.success("✅ All image-related tables dropped successfully.")


# --- Delete SpeciesNet Results File ---
if species_clicked:
    if SPECIESNET_RESULTS.exists():
        os.remove(SPECIESNET_RESULTS)
        st.success("✅ speciesnet_results.json deleted successfully.")
    else:
        st.warning("⚠️ speciesnet_results.json file not found.")


# --- Clear Data Lake Directories (RAW & JPG) ---
if datalake_clicked:
    # RAW Directory
    if RAW_DIR.exists() and RAW_DIR.is_dir():
        for file in RAW_DIR.glob("*"):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
        st.success("✅ RAW Data Lake path cleared successfully.")
    else:
        st.warning("⚠️ RAW Data Lake path not found.")

    # JPG Directory
    if JPG_DIR.exists() and JPG_DIR.is_dir():
        for file in JPG_DIR.glob("*"):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
        st.success("✅ JPG Data Lake path cleared successfully.")
    else:
        st.warning("⚠️ JPG Data Lake path not found.")
