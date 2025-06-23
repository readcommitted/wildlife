"""
project.py — Wildlife Vision System Overview & Documentation
-------------------------------------------------------------

This Streamlit module presents an interactive project overview for the Wildlife Image Processing
and Semantic Search System. It provides structured, expandable sections covering:

- Project goals and motivation
- System architecture and workflow
- Data science and AI components
- Key features and benefits
- User guide for system usage
- Current limitations
- Future development roadmap

This overview is intended to onboard users, stakeholders, and collaborators, providing technical
and non-technical audiences with insight into system capabilities and design principles.

Dependencies:
- Streamlit for UI rendering

Author: Matt Scardino
Project: Wildlife Vision System
"""

import streamlit as st

# --- Project Overview Section ---
with st.expander("🔍 Project Overview"):
    st.write("""
    This project is an automated wildlife image processing and semantic search system.
    It aims to streamline the management, classification, and retrieval of wildlife images, making it efficient for researchers, photographers, and enthusiasts.

    - **Why This Project?**
      - Manual tagging and organization of images is slow and error-prone.
      - Automatic species detection reduces the need for manual identification.
      - Semantic search allows finding images based on context rather than strict keywords.
      - User annotations can correct and enhance data quality.
      - Ad-Hoc Identification allows users to provide missing information.
      - This correction data can be used to improve the model (continuous learning).
    """)

# --- System Workflow Section ---
with st.expander("📊 System Workflow"):
    st.write("""
    1. **Image Ingestion:** Images are uploaded (JPEG/NEF), and EXIF metadata is extracted.
    2. **Image Processing:** Automated tagging based on date, time, and species detection (SpeciesNet).
    3. **Semantic Search:** Images are stored with vector embeddings for context-aware search.
    4. **User Interaction:** Annotate images, tag them, and clean up the dataset.
    5. **Annotation Benefits:** User-provided annotations can enhance model accuracy and serve as training data for future models.
    6. **Ad-Hoc Identification:** Allows users to manually tag images where species were not identified.
    """)

# --- Data Science, AI, and Engineering Section ---
with st.expander("🤖 Data Science, AI, and Engineering"):
    st.write("""
    - **Data Science:** Automated metadata extraction and tagging streamline data preparation.
    - **Artificial Intelligence:** SpeciesNet detects species in images using computer vision (AI).
    - **Semantic Search with AI:** Using pgvector embeddings, we transform image metadata and annotations into a format suitable for semantic search — blending NLP (natural language processing) and computer vision.
    - **Annotation-Driven Learning:** User annotations can improve model accuracy over time and provide labeled data for future model training.
    - **Software Engineering:** Streamlined, modular design with PostgreSQL for efficient data storage and retrieval.
    """)

# --- Key Features Section ---
with st.expander("✨ Key Features"):
    st.write("""
    - Automated metadata tagging (date, time, species).
    - Semantic search using PostgreSQL with pgvector.
    - Image management (display, annotate, delete, clean).
    - Extensible to new species and models.
    - Continuous model improvement through user annotations and Ad-Hoc.
    """)

# --- User Guide Section ---
with st.expander("📖 User Guide"):
    st.write("""
    1. Upload images for processing.
    2. View processed images and detected species.
    3. Use search to find images by context (not just keywords).
    4. Annotate images with custom notes or tags.
    5. Use Ad-Hoc for identifying unknown species.
    """)

# --- Current Limitations Section ---
with st.expander("⚠️ Current Limitations"):
    st.write("""
    - Species detection accuracy depends on the model (SpeciesNet).
    - Some images may lack EXIF metadata, affecting automatic tags.
    - Search may not be perfect for very obscure queries.
    - User-provided annotations can vary in quality.
    """)

# --- Future Plans Section ---
with st.expander("🚀 Future Plans"):
    st.write("""
    - Improve species detection with more advanced models.
    - Enhance search performance and accuracy.
    - Add support for more species and image formats.
    - Automate model retraining using collected annotation data.
    """)
