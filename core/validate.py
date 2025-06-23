"""
validate.py — Image Metadata Validation & Embedding Pipeline
-------------------------------------------------------------

This Streamlit module allows users to validate and enrich image metadata
after classification. It provides an interactive UI to:

- View images pending metadata updates
- Manually assign species labels, behavior notes, and tags
- Select known locations from a smart searchable dropdown
- Update geolocation fields based on location lookup
- Generate a semantic text embedding using OpenAI for each image
- Commit all updates to the PostgreSQL database

The embeddings support semantic search and downstream analysis.

Dependencies:
- Streamlit
- SQLAlchemy
- OpenAI client API (v1)
- PIL for image display

Author: Matt Scardino
Project: Wildlife Vision System
"""

import streamlit as st
from pathlib import Path
from PIL import Image
from db.db import SessionLocal
from db.image_model import ImageHeader, ImageLabel, ImageEmbedding
import time
from db.species_model import SpeciesFlattened
from db.location_model import LocationLookup
from tools.openai_utils import get_embedding
from datetime import datetime
from config.settings import MEDIA_ROOT


# --- Expandable Process Overview ---
with st.expander("Show/Hide Validation Process", expanded=False):
    st.header("Validation Process Overview")
    st.write(
        """
        This process allows updating species labels, behaviors, tags, and locations after classification.

        **Steps:**
        1. View pending images with incomplete metadata
        2. Manually assign species labels, behaviors, tags
        3. Select known locations from dropdown (auto-fills coordinates)
        4. Generate AI-based text embeddings for semantic search
        5. Commit all updates to the database
        """
    )


def get_season_from_date(date_obj: datetime) -> str:
    """
    Returns the season (spring, summer, fall, winter) based on month of date.
    """
    month = date_obj.month
    if month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "fall"
    else:
        return "winter"


def build_text_embedding(image_id, session):
    """
    Constructs and stores a semantic text embedding based on:
    - species label
    - location description
    - behavior notes
    - user-provided tags
    """
    image = session.query(ImageHeader).filter(ImageHeader.image_id == image_id).first()
    if not image:
        return

    species_label = session.query(ImageLabel.label_value).filter_by(image_id=image_id, label_type="user").scalar()
    location = image.location_description or ""
    behavior = image.behavior_notes or ""
    tags = ", ".join(image.tags or [])

    prompt = f"{species_label}. {location}. {behavior}. Tags: {tags}".strip()
    embedding = get_embedding(prompt)

    session.query(ImageEmbedding).filter_by(image_id=image_id).update({"text_embedding": embedding})


def smart_species_match(label_value, session):
    """
    Attempts to resolve a species label to a known species_id using fuzzy match.
    Returns species_id if matched, otherwise -1.
    """
    if not label_value:
        return None
    result = session.query(SpeciesFlattened).filter(
        SpeciesFlattened.common_name.ilike(f"%{label_value}%")
    ).first()
    return result.species_id if result else -1


def show_processed_images():
    """
    Streamlit UI to:
    - Display images pending metadata update
    - Provide species label, behavior, tag, location entry
    - Allow batch updates to selected images
    - Trigger text embedding generation
    """
    session = SessionLocal()
    image_records = (
        session.query(ImageHeader)
        .filter(ImageHeader.metadata_updated == False)
        .order_by(ImageHeader.capture_date.asc())
        .all()
    )

    species_options = session.query(SpeciesFlattened.common_name).order_by(SpeciesFlattened.common_name).all()
    locations = session.query(LocationLookup).order_by(LocationLookup.location_description).all()
    species_list = [s[0] for s in species_options if s[0]]

    if not image_records:
        st.info("No images pending metadata update.")
        return

    # Session State Initialization
    if "selected_images" not in st.session_state:
        st.session_state.selected_images = set()
    if "page" not in st.session_state:
        st.session_state.page = 0
    if "per_page" not in st.session_state:
        st.session_state.per_page = 10
    if "metadata_fields" not in st.session_state:
        st.session_state.metadata_fields = {
            "species_label": "",
            "behavior_notes": "",
            "tags": "",
            "park": "",
            "place": ""
        }

    start = st.session_state.page * st.session_state.per_page
    end = start + st.session_state.per_page
    page_records = image_records[start:end]

    # Metadata Entry Form
    with st.form("validation_form", clear_on_submit=False):
        species_label = st.selectbox(
            "Species Label (smart search)",
            options=[""] + species_list,
            index=0,
            help="Start typing to search for a species by common name."
        )
        behavior_notes = st.text_input("Behavior Notes", value=st.session_state.metadata_fields["behavior_notes"])
        tags = st.text_input("Tags (comma-separated)", value=st.session_state.metadata_fields["tags"])

        location_options = [l.location_description for l in locations if l.location_description]
        selected_location = st.selectbox(
            "Location Description (Smart Search)",
            [""] + location_options,
            help="Choose a known location from the list"
        )

        submit = st.form_submit_button("✅ Update Selected Images")

    # Pagination Controls
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("⏮️ Previous", disabled=st.session_state.page == 0):
            st.session_state.page -= 1
    with col2:
        if st.button("Next ⏭️", disabled=end >= len(image_records)):
            st.session_state.page += 1

    # Display Settings
    st.markdown("### Display Settings")
    col1, col2 = st.columns(2)
    with col1:
        preview_size = st.radio("Thumbnail Size", ["Small", "Large"], horizontal=True)
        thumb_width = 90 if preview_size == "Small" else 150

    # Image Grid
    for record in page_records:
        img_path = Path(record.jpeg_path)
        if not img_path.exists():
            continue

        species = session.query(SpeciesFlattened).filter_by(species_id=record.species_id).first()
        common_name = species.common_name if species else "Unknown"

        with st.container():
            cols = st.columns([1, 4, 2])
            with cols[0]:
                img = Image.open(img_path)
                st.image(img, width=thumb_width)

            with cols[1]:
                st.markdown(f"**{record.image_name}**  \nSpecies: *{common_name}*  \nLocation: *{record.location_description or 'N/A'}*")

            with cols[2]:
                default_selected = record.image_id in st.session_state.selected_images
                selected = st.checkbox("Select", value=default_selected, key=f"check_{record.image_id}")
                if selected:
                    st.session_state.selected_images.add(record.image_id)
                else:
                    st.session_state.selected_images.discard(record.image_id)

    # --- Form Submission Handler ---
    if submit:
        try:
            matched_species_id = smart_species_match(species_label.strip(), session) if species_label.strip() else -1
            loc = session.query(LocationLookup).filter_by(location_description=selected_location).first() if selected_location else None

            for image_id in st.session_state.selected_images:
                if species_label.strip():
                    session.add(ImageLabel(
                        image_id=image_id,
                        label_type="user",
                        label_value=species_label.strip(),
                        label_source="user",
                        confidence=1.0
                    ))

                image = session.query(ImageHeader).filter(ImageHeader.image_id == image_id).first()
                tag_list = tags.split(",") if tags else []

                # Add season tag
                if image.capture_date:
                    season = get_season_from_date(image.capture_date)
                    season_tag = f"season: {season}"
                    if season_tag not in tag_list:
                        tag_list.append(season_tag)
                    if behavior_notes:
                        if f"Observed in {season}" not in behavior_notes:
                            behavior_notes += f" Observed in {season}."
                    else:
                        behavior_notes = f"Observed in {season}."

                update_fields = {
                    ImageHeader.metadata_updated: True,
                    ImageHeader.behavior_notes: behavior_notes or None,
                    ImageHeader.tags: tag_list
                }

                if loc:
                    update_fields.update({
                        ImageHeader.location_description: loc.location_description,
                        ImageHeader.park: loc.park,
                        ImageHeader.place: loc.place,
                        ImageHeader.state_code: loc.state,
                        ImageHeader.country_code: loc.country,
                        ImageHeader.county: loc.county
                    })
                    if (not image.latitude or image.latitude == 0) and (not image.longitude or image.longitude == 0):
                        update_fields.update({
                            ImageHeader.latitude: loc.latitude,
                            ImageHeader.longitude: loc.longitude
                        })

                session.query(ImageHeader).filter(ImageHeader.image_id == image_id).update(update_fields)
                build_text_embedding(image_id, session)

            session.commit()
            st.success(f"✅ Updated {len(st.session_state.selected_images)} image(s)")

            st.session_state.selected_images.clear()
            st.session_state.metadata_fields = {
                "species_label": "",
                "behavior_notes": "",
                "tags": "",
                "park": "",
                "place": ""
            }
            time.sleep(1)
            st.rerun()

        except Exception as e:
            session.rollback()
            st.error(f"Database update failed: {e}")
        finally:
            session.close()


# Run the validation interface
show_processed_images()
