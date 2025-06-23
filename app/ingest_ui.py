"""
ingest.py — Image Ingestion & Preprocessing UI
------------------------------------------------

This Streamlit module handles the ingestion and preprocessing of RAW wildlife images
prior to classification and analysis. It provides an interactive UI to:

- Monitor staged RAW images pending processing
- Execute the ingestion pipeline with visual progress updates
- Clear previous results and reset the staging environment
- Display an expandable process overview for transparency

Ingestion Pipeline Includes:
1. EXIF Metadata extraction from RAW (NEF) files
2. Capture date parsing and directory organization
3. RAW to JPEG conversion using rawpy and OpenCV
4. Saving processed files to structured data lake directories
5. Metadata insertion into the PostgreSQL database

Species Identification Pipeline Includes:
- Running the SpeciesNet model on processed images
- Storing predictions in JSON and updating the database

Dependencies:
- Streamlit for UI
- rawpy, OpenCV for image processing (handled in core.ingest)
- exiftool (external) for metadata extraction
- PostgreSQL backend
- SpeciesNet model for classification

Author: Matt Scardino
Project: Wildlife Vision System
"""

import streamlit as st
import shutil
from config.settings import STAGE_DIR, STAGE_PROCESSED_DIR, PREDICTIONS_JSON


# --- Expandable Process Overview UI ---
with st.expander("Show/Hide Ingestion Process Overview", expanded=False):
    st.header("1. Ingestion Process")
    st.write(
        """
        This process is responsible for processing RAW image files (NEF format) and preparing them for analysis.

        **Steps:**
        1. **Extract Metadata**: Uses `exiftool` to extract EXIF metadata from the RAW files.
        2. **Extract Capture Date**: Uses the metadata to get the image capture date.
        3. **Create Data Lake Directories**: Organizes images by date in a structured folder format.
        4. **Convert RAW to JPEG**:
           - Uses `rawpy` for RAW image reading.
           - Uses `OpenCV` for JPEG conversion and saving.
        5. **Move Files to Data Lake**: Saves the processed JPEG and RAW files in their respective directories.
        6. **Insert Metadata into Database**: Saves image metadata (including EXIF) to the PostgreSQL database.
        """
    )

    st.header("2. Species Identification Process")
    st.write(
        """
        This process is dedicated to running the SpeciesNet model on the processed images (JPEG).

        **Steps:**
        1. **Load Processed Images**: Reads all JPEG images from the `stage_processed` directory.
        2. **Run SpeciesNet Model**:
           - Uses the `speciesnet.scripts.run_model` script.
           - The model analyzes the images and generates species predictions.
        3. **Update Results**:
           - Saves predictions to a JSON file (`speciesnet_results.json`).
           - Updates species information in the PostgreSQL database.
        """
    )

    st.header("3. Models Used")
    st.write(
        """
        - **SpeciesNet Model**: Custom model for species identification.
        - **EXIF Metadata Extraction**: Uses `exiftool` for metadata extraction from RAW images.
        - **Image Processing**: Uses `rawpy` for RAW file decoding and `OpenCV` for image processing.
        """
    )

# --- Progress Log Helper Function ---
def update_status(message):
    """
    Append a status message to the progress log and display recent entries.
    """
    if "progress_log" not in st.session_state:
        st.session_state.progress_log = []

    st.session_state.progress_log.append(message)

    progress_area.markdown(
        "<div style='height: 350px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px;'>"
        + "<br>".join(st.session_state.progress_log[-15:])  # Show last 15 messages
        + "</div>",
        unsafe_allow_html=True
    )

# --- Step 1: Check for Staged RAW Files ---
raw_files = [f for f in STAGE_DIR.glob("*") if f.suffix.lower() == ".nef"]

if raw_files:
    st.info(f"{len(raw_files)} images ready for processing.")
else:
    st.warning("No RAW images found in staging.")

# --- Action Buttons ---
col1, col2 = st.columns([1, 1])

with col1:
    clear_clicked = st.button("Clear Results")

with col2:
    process_clicked = st.button("Process Images")

# --- Step 2: Button Action Handling ---
progress_area = st.empty()

if "progress_log" not in st.session_state:
    st.session_state.progress_log = []

if clear_clicked:
    progress_area.empty()
    if PREDICTIONS_JSON.exists():
        PREDICTIONS_JSON.unlink()
    shutil.rmtree(STAGE_PROCESSED_DIR, ignore_errors=True)
    progress_area.success("Processing results cleared.")

elif process_clicked:
    progress_area.empty()
    import core.ingest  # Dynamic import to avoid circular dependencies during initial load

    st.session_state.progress_text = "Starting ingestion..."
    progress_area.text(st.session_state.progress_text)

    # Run ingestion with real-time status updates
    core.ingest.process_raw_images(update_status)

    progress_area.empty()
    progress_area.success("Images processed and metadata written to database.")
