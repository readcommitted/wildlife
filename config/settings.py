"""
settings.py - Central configuration for Wildlife Image Processing & Semantic Search System

All file system paths are derived from MEDIA_ROOT to ensure consistent structure.
Secrets (API keys, tokens) reside in secrets.toml for Streamlit.
"""

import os
from dotenv import load_dotenv
from pathlib import Path


# --- Load .env ---

load_dotenv()


# --- Helper Functions ---

def normalize_root(env_var: str, default: str) -> Path:
    """
    Resolve a filesystem path from an environment variable, falling back to a default.
    """
    raw = os.getenv(env_var, default)
    if not (raw.startswith(os.sep) or raw.startswith('.') or raw.startswith('~')):
        raw = os.sep + raw
    return Path(raw).expanduser().resolve()


# --- Core Paths ---

MEDIA_ROOT = normalize_root('MEDIA_ROOT', './media')

# Derived structure based on MEDIA_ROOT
STAGE_DIR = MEDIA_ROOT / "stage"
STAGE_PROCESSED_DIR = MEDIA_ROOT / "stage_processed"
RAW_DIR = MEDIA_ROOT / "raw"
JPG_DIR = MEDIA_ROOT / "jpg"
STATIC_ROOT = MEDIA_ROOT / "static"
PREDICTIONS_JSON = MEDIA_ROOT / os.getenv('PREDICTIONS_JSON', 'speciesnet_results.json')
IMAGE_DIR = MEDIA_ROOT / "species_images"

# --- Database & Environment ---

DATABASE_URL = os.getenv('DATABASE_URL')
DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')
ENVIRONMENT = os.getenv('ENV', 'development')
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv('DEFAULT_CONFIDENCE_THRESHOLD', 0.6))

# --- Wikipedia & API Config ---
WIKI_API_URL = os.getenv('WIKI_API_URL', 'https://en.wikipedia.org/w/api.php')
USER_AGENT = os.getenv('USER_AGENT', 'WildlifeImageBot/1.0 (example@example.com)')
HEADERS = {
    "User-Agent": USER_AGENT
}

# --- Secrets & User-Agent ---
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
    USER_AGENT = st.secrets.get("USER_AGENT", os.getenv('USER_AGENT', 'WildlifeImageBot/1.0 (example@example.com)'))
except ImportError:
    OPENAI_API_KEY = None
    USER_AGENT = os.getenv('USER_AGENT', 'WildlifeImageBot/1.0 (example@example.com)')
