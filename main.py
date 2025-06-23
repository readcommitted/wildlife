"""
main.py — Streamlit Multi-Page Navigation Controller
-----------------------------------------------------

This script initializes the Wildlife Vision System Streamlit app and
provides dynamic, config-driven page navigation.

Features:
✅ Supports both flat and grouped sidebar navigation
✅ Loads pages from `.streamlit/pages.toml` (flat) or `pages_sections.toml` (grouped)
✅ Automatically sets the page title and browser tab icon
✅ Python 3.12 compatibility patch for event loops

Navigation and layout logic are driven by `st_pages` extension.

Dependencies:
- streamlit
- st_pages

Author: Matt Scardino
Project: Wildlife Vision System
"""

import streamlit as st
from st_pages import add_page_title, get_nav_from_toml
import asyncio

# Patch for Python 3.12 compatibility with Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Configure the main Streamlit app window ---
st.set_page_config(
    page_title="Wildlife Image Processing & Semantic Search System",
    page_icon="🐾",               # Tab icon
    layout="wide",                # Use full width of the browser
    initial_sidebar_state="expanded"
)

# --- Sidebar toggle to choose between flat pages or grouped sections ---
sections = st.sidebar.toggle(
    "Sections",
    value=True,
    key="use_sections"
)

# --- Load navigation config from the appropriate TOML file ---
nav = get_nav_from_toml(
    ".streamlit/pages_sections.toml" if sections else ".streamlit/pages.toml"
)

# --- Create the navigation sidebar ---
pg = st.navigation(nav)

# --- Automatically display page title from TOML definition ---
add_page_title(pg)

# --- Run the selected page from the sidebar ---
pg.run()
