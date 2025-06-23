"""
ingest.py — Image Ingestion, Processing, and Metadata Pipeline
---------------------------------------------------------------

This module handles the ingestion and preprocessing of RAW wildlife images.
It extracts metadata, enriches geolocation information, converts images to JPEG,
detects subjects using YOLO, generates CLIP embeddings, and inserts records into the database.

Key Features:
- EXIF metadata extraction using `exiftool`
- RAW to JPEG conversion with `rawpy` and `OpenCV`
- Automatic animal detection and cropping via YOLO
- Image embedding generation with CLIP (ViT-B/32)
- Database insertion of structured image, metadata, and embedding records
- Organized image storage in a data lake hierarchy by date

Dependencies:
- rawpy, OpenCV, PIL for image handling
- exiftool (external) for metadata extraction
- YOLO for animal detection
- CLIP for image embedding generation
- SQLAlchemy for database operations

Author: Matt Scardino
Project: Wildlife Vision System
"""

from pathlib import Path
import subprocess
import json
import shutil
from datetime import datetime
import cv2
import numpy as np
import rawpy
import torch
import clip
import re
from PIL import Image
from tools.embedding_utils import generate_image_embedding
from tools.geo_utils import get_location_metadata
from tools.yolo_detector import YOLODetector
from db.db import SessionLocal
from db.image_model import ImageHeader, ImageExif, ImageEmbedding
from config.settings import MEDIA_ROOT, STAGE_DIR, STAGE_PROCESSED_DIR, RAW_DIR, JPG_DIR


# --- Configuration ---
EXPOSURE_ADJUSTMENT = 1.6  # Brightness adjustment for RAW to JPEG conversion

# Load CLIP Model for Image Embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def dms_to_decimal(dms_str: str) -> float:
    """
    Convert GPS coordinates from DMS (degrees, minutes, seconds) to decimal format.
    """
    match = re.match(r"(\d+) deg (\d+)' ([\d.]+)\" ([NSEW])", dms_str)
    if not match:
        raise ValueError(f"Invalid DMS string format: {dms_str}")
    degrees, minutes, seconds, direction = match.groups()
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ['S', 'W']:
        decimal *= -1
    return decimal


def parse_exposure_compensation(value):
    """
    Safely parse exposure compensation from EXIF metadata.
    Handles fraction formats or raw float values.
    """
    if isinstance(value, str) and "/" in value:
        try:
            numerator, denominator = value.replace("+", "").replace("-", "-").split("/")
            return float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            pass
    return float(value) if value else None


def extract_nef_exif_with_exiftool(file_path):
    """
    Extract EXIF metadata from a NEF file using exiftool.
    Returns parsed JSON dictionary or None on failure.
    """
    try:
        result = subprocess.run([
            "exiftool",
            "-json",
            str(file_path)
        ], capture_output=True, text=True)
        metadata = json.loads(result.stdout)[0] if result.returncode == 0 else None
        return metadata
    except Exception as e:
        print(f"❌ Error extracting EXIF with exiftool from {file_path}: {e}")
        return None


def convert_nef_to_jpeg(raw_file):
    """
    Convert NEF RAW file to JPEG, apply YOLO animal detection and cropping.
    Saves JPEG to staging directory and returns path.
    """
    STAGE_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    jpeg_path = STAGE_PROCESSED_DIR / f"{raw_file.stem}.jpg"

    try:
        with rawpy.imread(str(raw_file)) as raw:
            rgb_image = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
                bright=EXPOSURE_ADJUSTMENT
            )
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        pil_image = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))

        detector = YOLODetector()
        detected_crops = detector.detect_and_crop(pil_image)

        if not detected_crops:
            print("No animals detected by YOLO. Falling back to full image.")
            detected_crops = [pil_image]
        else:
            print(f"Detected {len(detected_crops)} animal(s) with YOLO.")

        first_crop = detected_crops[0]
        np_crop = cv2.cvtColor(np.array(first_crop), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(jpeg_path), np_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        return jpeg_path if jpeg_path.exists() else None

    except Exception as e:
        print(f"❌ Error converting {raw_file} to JPEG with rawpy and OpenCV: {e}")
        return None


def insert_metadata_to_db(metadata, raw_data_lake_path, jpeg_data_lake_path, raw_file, update_status):
    """
    Insert image metadata, location details, and image embedding into the database.
    """
    session = SessionLocal()

    capture_date = metadata.get("SubSecCreateDate")
    if not capture_date:
        print(f"❌ No capture date found in EXIF for {raw_file.name}. Skipping.")
        return

    clean_capture_date = capture_date.split(".")[0].split("-")[0]
    capture_date_obj = datetime.strptime(clean_capture_date, "%Y:%m:%d %H:%M:%S")

    lat, lon = metadata.get("GPSLatitude"), metadata.get("GPSLongitude")
    if lat and lon:
        loc = get_location_metadata(dms_to_decimal(lat), dms_to_decimal(lon))
        lat_val = loc.get("latitude")
        lon_val = loc.get("longitude")
        location_desc = loc.get("full_display_name")
    else:
        lat_val, lon_val = 0.0, 0.0
        location_desc = "Unknown"
        loc = {}

    pre_embedding = generate_image_embedding(jpeg_data_lake_path)
    update_status(f"Pre-Embedding Generated (512-D): {jpeg_data_lake_path}")

    image_header = ImageHeader(
        image_name=raw_file.name,
        raw_path=str(raw_data_lake_path),
        jpeg_path=str(jpeg_data_lake_path),
        stage_processed_path=str(STAGE_PROCESSED_DIR / f"{raw_file.stem}.jpg"),
        capture_date=capture_date_obj,
        current_batch=True,
        latitude=lat_val,
        longitude=lon_val,
        location_description=location_desc,
        country_code=loc.get("country"),
        state_code=loc.get("state"),
        county=loc.get("county"),
        park=loc.get("park"),
        place=loc.get("place"),
        metadata_updated=False,
    )

    image_exif = ImageExif(
        photographer=metadata.get("Artist"),
        camera_model=metadata.get("Model"),
        lens_model=metadata.get("LensModel"),
        focal_length=metadata.get("FocalLength"),
        exposure_time=metadata.get("ExposureTime"),
        aperture=metadata.get("FNumber"),
        iso=metadata.get("ISO"),
        shutter_speed=metadata.get("ShutterSpeed"),
        exposure_program=metadata.get("ExposureProgram"),
        exposure_compensation=parse_exposure_compensation(metadata.get("ExposureCompensation")),
        metering_mode=metadata.get("MeteringMode"),
        light_source=metadata.get("LightSource"),
        white_balance=metadata.get("WhiteBalance"),
        flash=metadata.get("Flash"),
        color_space=metadata.get("ColorSpace"),
        subject_detection=metadata.get("SubjectDetection"),
        autofocus=metadata.get("AutoFocus"),
        serial_number=metadata.get("SerialNumber"),
        software_version=metadata.get("Software"),
        exif_json=json.dumps(metadata)
    )

    image_embedding = ImageEmbedding(
        image_embedding=pre_embedding.tolist()
    )

    image_header.image_exif = image_exif
    image_header.image_embedding = image_embedding
    session.add(image_header)
    session.commit()
    session.close()


def process_raw_images(update_status):
    """
    Full ingestion pipeline:
    - Processes new RAW images from staging
    - Extracts metadata and generates JPEG
    - Runs YOLO detection and CLIP embedding
    - Moves files to organized data lake directories
    - Inserts metadata and embedding into database
    """
    raw_files = [f for f in STAGE_DIR.iterdir() if f.suffix.lower() == ".nef" and f.is_file()]
    if not raw_files:
        update_status("❌ No RAW images found.")
        return

    with SessionLocal() as session:
        existing_names = {name for (name,) in session.query(ImageHeader.image_name).all()}
    raw_files = [f for f in raw_files if f.name not in existing_names]

    update_status(f"Found {len(raw_files)} RAW images in staging.")

    # Reset current_batch flag
    session = SessionLocal()
    session.query(ImageHeader).update({ImageHeader.current_batch: False})
    session.commit()
    session.close()

    total_files = len(raw_files)
    for index, raw_file in enumerate(raw_files, start=1):
        update_status(f"Processing {raw_file.name}, {index} of {total_files}...")

        metadata = extract_nef_exif_with_exiftool(raw_file)
        if not metadata:
            update_status(f"❌ Failed to extract metadata for {raw_file.name}")
            continue

        capture_date = metadata.get("SubSecCreateDate")
        if not capture_date:
            update_status(f"❌ No capture date found for {raw_file.name}. Skipping.")
            continue

        try:
            capture_date_obj = datetime.strptime(capture_date.split()[0], "%Y:%m:%d")
            date_path = capture_date_obj.strftime("%Y/%m/%d")
        except ValueError:
            update_status(f"❌ Invalid capture date format for {raw_file.name}. Skipping.")
            continue

        raw_output_path = RAW_DIR / date_path
        jpg_output_path = JPG_DIR / date_path
        raw_output_path.mkdir(parents=True, exist_ok=True)
        jpg_output_path.mkdir(parents=True, exist_ok=True)

        jpeg_path = convert_nef_to_jpeg(raw_file)
        if not jpeg_path or not jpeg_path.exists():
            update_status(f"❌ Failed to convert {raw_file.name} to JPEG.")
            continue

        shutil.copy(raw_file, raw_output_path / raw_file.name)
        shutil.copy(jpeg_path, jpg_output_path / jpeg_path.name)
        update_status(f"Moved files to data lake for {raw_file.name}")

        insert_metadata_to_db(
            metadata,
            raw_output_path / raw_file.name,
            jpg_output_path / jpeg_path.name,
            raw_file,
            update_status
        )
        update_status(f"Metadata written to database for {raw_file.name}")

    update_status("✅ All images processed.")
