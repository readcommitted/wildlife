"""
analysis_ui.py — Species Analysis & Embedding Validation Dashboard
-------------------------------------------------------------------

This Streamlit module provides interactive tools for validating species predictions,
comparing regional embedding performance, and visualizing the embedding space.

Features:
- SpeciesNet Validation: Evaluate predicted species labels vs. user-labeled ground truth
- CLIP Validation: Compare CLIP semantic predictions with manual species labels
- CLIP Region Comparison: Visualize similarity of species within a target ecoregion
- UMAP Projection: Explore species clustering using dimensionality reduction

Supports:
- Ad-hoc validation and performance monitoring of SpeciesNet and CLIP models
- Geographic comparisons based on known ecoregions
- Intuitive, interactive UI for model evaluation

Dependencies:
- Streamlit for UI controls
- Core analysis utilities (speciesnet, CLIP validation, UMAP)
- SQLAlchemy models for database access

Author: Matt Scardino
Project: Wildlife Vision System
"""

import streamlit as st
from core.analysis import (
    run_clip_label_vs_prediction_analysis,
    display_umap_species_projection,
    speciesnet,
    clip_region_comparison
)


# --- Sidebar Option: Limit Number of Images for Analysis ---
limit = st.sidebar.slider("Number of Images", 10, 500, 200)

# --- Main Tab Layout for Analysis Tools ---
tab1, tab2, tab3, tab4 = st.tabs([
    "SpeciesNet Validation",
    "Clip Validation",
    "Clip Region Comparison",
    "UMAP Projection"
])

# --- Tab 1: SpeciesNet Validation ---
with tab1:
    st.subheader("📄 SpeciesNet Evaluation")
    if st.button("Run SpeciesNet Evaluation"):
        speciesnet(limit)

# --- Tab 2: CLIP Label vs Prediction Analysis ---
with tab2:
    st.subheader("📄 CLIP Embedding Validation")
    if st.button("Run CLIP Label vs Prediction Analysis"):
        run_clip_label_vs_prediction_analysis()

# --- Tab 3: CLIP Region-Specific Comparison ---
with tab3:
    st.subheader("🗺️ CLIP Region Comparison – South Central Rockies Forests")
    clip_region_comparison(limit)

# --- Tab 4: UMAP Species Embedding Projection ---
with tab4:
    display_umap_species_projection()
