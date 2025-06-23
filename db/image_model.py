"""
image_model.py — Wildlife Image ORM Models
-------------------------------------------

This module defines the SQLAlchemy ORM models for managing image metadata, EXIF data,
embeddings, and user labels in the Wildlife Vision System.

Tables:
- `wildlife.image_header`: Core image metadata and location details
- `wildlife.image_exif`: EXIF metadata extracted from RAW images
- `wildlife.image_embedding`: CLIP and semantic embeddings for search and analysis
- `wildlife.image_label`: User-provided or automated labels for species and context

Features:
- One-to-one relationships between image, EXIF, and embeddings
- Support for semantic tags, behavior notes, and geographic enrichment
- PgVector fields for image, text, and hybrid embeddings

Dependencies:
- SQLAlchemy ORM
- pgvector extension for vector similarity search

Author: Matt Scardino
Project: Wildlife Vision System
"""

from sqlalchemy import Column, Integer, Float, Text, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ImageHeader(Base):
    """
    Table: wildlife.image_header

    Stores core image metadata including:
    - File paths
    - Capture date
    - Species detection results
    - Location details (GPS and OpenStreetMap)
    - Behavior notes and user-assigned tags

    Relationships:
    - One-to-one with ImageExif (EXIF data)
    - One-to-one with ImageEmbedding (vector embeddings)
    """
    __tablename__ = "image_header"
    __table_args__ = {"schema": "wildlife"}

    image_id = Column(Integer, primary_key=True)

    image_exif = relationship("ImageExif", uselist=False, back_populates="image_header", cascade="all, delete-orphan")
    image_embedding = relationship("ImageEmbedding", uselist=False, back_populates="image_header", cascade="all, delete-orphan")

    image_name = Column(Text, nullable=False)
    raw_path = Column(Text)
    jpeg_path = Column(Text)
    stage_processed_path = Column(Text)
    capture_date = Column(DateTime)
    current_batch = Column(Boolean, default=False)

    species_id = Column(Integer, nullable=False, default=-1)
    speciesnet_raw = Column(Text)
    species_confidence = Column(Float)
    species_detection_method = Column(Text)

    latitude = Column(Float)
    longitude = Column(Float)
    osm_latitude = Column(Float)
    osm_longitude = Column(Float)
    location_description = Column(Text)
    country_code = Column(Text)
    state_code = Column(Text)
    county = Column(Text)
    park = Column(Text)
    place = Column(Text)

    metadata_updated = Column(Boolean, default=False)
    behavior_notes = Column(Text)
    tags = Column(ARRAY(Text))

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())


class ImageExif(Base):
    """
    Table: wildlife.image_exif

    Stores EXIF metadata extracted from RAW image files, including:
    - Camera and lens information
    - Exposure, aperture, ISO
    - Full raw EXIF JSON for reference
    """
    __tablename__ = "image_exif"
    __table_args__ = {"schema": "wildlife"}

    image_exif_id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("wildlife.image_header.image_id"), nullable=False)

    image_header = relationship("ImageHeader", back_populates="image_exif")

    photographer = Column(Text)
    camera_model = Column(Text)
    lens_model = Column(Text)
    focal_length = Column(Text)
    exposure_time = Column(Text)
    aperture = Column(Text)
    iso = Column(Integer)
    shutter_speed = Column(Text)
    exposure_program = Column(Text)
    exposure_compensation = Column(Float)
    metering_mode = Column(Text)
    light_source = Column(Text)
    white_balance = Column(Text)
    flash = Column(Text)
    color_space = Column(Text)
    subject_detection = Column(Text)
    autofocus = Column(Text)
    serial_number = Column(Text)
    software_version = Column(Text)
    exif_json = Column(JSON)


class ImageEmbedding(Base):
    """
    Table: wildlife.image_embedding

    Stores vector embeddings for each image, including:
    - CLIP image embedding (512-dim)
    - CLIP image+text hybrid embedding (512-dim)
    - OpenAI text embedding for semantic search (1536-dim)
    - Species similarity score

    Supports downstream search, clustering, and species validation tasks.
    """
    __tablename__ = "image_embedding"
    __table_args__ = {"schema": "wildlife"}

    image_embedding_id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("wildlife.image_header.image_id"), nullable=False)

    image_header = relationship("ImageHeader", back_populates="image_embedding")

    image_embedding = Column(Vector(512))
    image_text_embedding = Column(Vector(512))
    text_embedding = Column(Vector(1536))
    embedding_method = Column(Text)
    embedding_date = Column(DateTime, server_default=func.now())
    common_name = Column(Text)
    score = Column(Float)


class ImageLabel(Base):
    """
    Table: wildlife.image_label

    Stores user or automated labels for images, including:
    - Label type (e.g., "user", "model", "annotation")
    - Label value (species name, behavior, etc.)
    - Label source and confidence
    """
    __tablename__ = 'image_label'
    __table_args__ = {'schema': 'wildlife'}

    image_label_id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('wildlife.image_header.image_id'))

    label_type = Column(Text)
    label_value = Column(Text)
    label_source = Column(Text)
    confidence = Column(Float)
    created_at = Column(DateTime, server_default=func.now())
