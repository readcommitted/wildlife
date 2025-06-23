"""
regenerate_embeddings_ui.py — Embedding Maintenance Utilities
--------------------------------------------------------------

This Streamlit module provides utility functions to maintain and regenerate
embeddings used within the Wildlife Vision System.

Features:
- Regenerate all image embeddings for visual similarity and semantic search
- Recompute species embeddings for canonical species vectors
- Populate missing common names in the image embedding table

These tools are intended for maintenance, debugging, or batch updates when:
- Embedding models are retrained
- Database inconsistencies are detected
- Missing or incomplete species metadata needs correction

Dependencies:
- Streamlit for UI controls
- regenerate_image_embeddings module for backend processing

Author: Matt Scardino
Project: Wildlife Vision System
"""

import streamlit as st
from tools.regenerate_image_embeddings import (
    regenerate_image_embeddings_and_score,
    regenerate_species_embeddings,
    update_missing_common_names
)

# --- Streamlit UI Header ---
st.header("🧠 Embedding Utilities")

# --- Action Buttons Layout ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    image_clicked = st.button("Regenerate All Image Embeddings")

with col2:
    species_clicked = st.button("Regenerate All Species Embeddings")

with col3:
    update_common = st.button("Generate Common Names (Missing Only)")

# --- Action Handlers ---
if image_clicked:
    regenerate_image_embeddings_and_score()

if species_clicked:
    regenerate_species_embeddings()

if update_common:
    update_missing_common_names()
