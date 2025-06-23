"""
embedding_utils.py — Image Embedding Generator (CLIP Model)
------------------------------------------------------------

This module provides utilities to generate image embeddings using OpenAI's CLIP model (ViT-B/32),
used for species identification, semantic search, and similarity comparisons within the Wildlife Vision System.

Features:
- Loads CLIP model (GPU if available, fallback to CPU)
- Generates normalized 512-dimensional image embeddings
- Supports standardized preprocessing for all images

Dependencies:
- PyTorch for tensor operations
- PIL for image handling
- OpenAI CLIP model

Author: Matt Scardino
Project: Wildlife Vision System
"""

import torch
from PIL import Image
import clip

# --- Device and Model Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def generate_image_embedding(image_path):
    """
    Generates a normalized 512-dimensional embedding for a given image.

    Args:
        image_path (str or Path): Path to the image file.

    Returns:
        np.ndarray: Flattened 512-dim image embedding vector.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(image_tensor)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        image_embedding = image_embedding.cpu().numpy().flatten()

    return image_embedding
