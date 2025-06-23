"""
analysis.py — Wildlife Species Embedding Analysis Tools
-------------------------------------------------------

This module provides visualization and evaluation tools for:

✅ Comparing CLIP-predicted species against user-provided labels
✅ Visualizing species embeddings with UMAP projections
✅ Analyzing SpeciesNet detection accuracy and confidence
✅ Exploring species similarity within ecological regions

Uses:
- CLIP (ViT-B/32) for image/text embeddings
- PostgreSQL (via SQLAlchemy) for image/label data
- Plotly and Matplotlib for visualizations

Author: Matt Scardino
Project: Wildlife Vision System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import umap
import torch
import clip
import ast
from matplotlib import pyplot as plt
from sqlalchemy import text
from db.db import SessionLocal
from db.image_model import ImageEmbedding, ImageLabel, ImageHeader
from db.species_model import SpeciesFlattened


# --- CLIP Model Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def run_clip_label_vs_prediction_analysis():
    """Compare CLIP predicted species with user-provided labels (image-level)"""
    with SessionLocal() as session:
        rows = session.query(
            ImageEmbedding.image_id,
            ImageEmbedding.common_name,
            ImageEmbedding.score,
            ImageLabel.label_value
        ).join(ImageLabel, ImageEmbedding.image_id == ImageLabel.image_id) \
         .filter(ImageLabel.label_type == 'user') \
         .all()

    results = []
    for image_id, predicted, score, true_label in rows:
        agreement = (
            predicted and true_label and
            predicted.lower().strip() == true_label.lower().strip()
        )
        results.append({
            "image_id": image_id,
            "label": true_label,
            "predicted": predicted,
            "confidence score": round(score * 100, 2) if score is not None else None,
            "match": "✅" if agreement else "❌"
        })

    df = pd.DataFrame(results).sort_values(by="image_id", ascending=False)
    st.dataframe(df[["image_id", "label", "predicted", "confidence score", "match"]])

    correct = sum(r["match"] == "✅" for r in results)
    total = len(results)
    st.markdown(f"### ✅ Accuracy: {correct} / {total} = **{correct / total:.2%}**")


def display_umap_species_projection():
    """UMAP 2D projection of species and image embeddings with drift detection"""
    st.subheader("📊 UMAP Species Embedding Visualization")

    ECOREGION_CODES = ["NA0528"]  # Yellowstone + Colorado
    ECOREGION_NAME = "Colorado + Yellowstone"

    with SessionLocal() as session:
        st.markdown("#### ℹ️ About this view")
        with st.expander("What is this?"):
            st.markdown("""
            Visualizes species embeddings from the CLIP model:

            - **Blue**: Canonical species vectors  
            - **Gold**: Canonical species used in image labels  
            - **Green**: Image predictions matching user label  
            - **Red**: Image predictions differing from user label  
            """)

        class_filter = st.selectbox("Filter by Class", ["mammals", "birds"])
        show_overlay = st.checkbox("Overlay image-level embeddings", value=True)

        species_data = session.execute(text("""
            SELECT DISTINCT e.common_name, e.category, e.image_embedding
            FROM wildlife.species_embedding e
            JOIN wildlife.species_by_region r ON e.common_name = r.common_name
            WHERE r.ecoregion_code = ANY(:regions)
              AND e.category = :cls
              AND e.image_embedding IS NOT NULL
        """), {"regions": ECOREGION_CODES, "cls": class_filter}).fetchall()

        if not species_data:
            st.warning(f"No species embeddings found for '{class_filter}' in {ECOREGION_NAME}")
            st.stop()

        species_names, species_vecs = [], []
        for name, _, emb in species_data:
            emb = ast.literal_eval(emb) if isinstance(emb, str) else emb
            species_names.append(name)
            species_vecs.append(emb)
        species_vecs = np.array(species_vecs)

        labeled_species = {r[0] for r in session.query(ImageLabel.label_value.distinct())
                           .filter(ImageLabel.label_type == 'user').all()}

        image_species = [r[0] for r in session.query(ImageEmbedding.common_name.distinct())
                         .filter(ImageEmbedding.common_name.in_(species_names)).all()]

        label_filter = st.selectbox("Filter Image Embeddings by Species", ["All"] + image_species)

        # Prepare vectors and labels
        all_vecs = list(species_vecs)
        all_labels = ["canonical species vectors"] * len(species_vecs)
        all_names = species_names.copy()
        all_paths = [None] * len(species_vecs)
        all_ids = [None] * len(species_vecs)
        all_true = [None] * len(species_vecs)
        all_scores = [None] * len(species_vecs)

        # Overlay image embeddings if selected
        if show_overlay:
            query = session.query(
                ImageEmbedding.image_id,
                ImageEmbedding.image_embedding,
                ImageEmbedding.common_name,
                ImageHeader.jpeg_path,
                ImageEmbedding.score,
                ImageLabel.label_value
            ).join(ImageHeader, ImageEmbedding.image_id == ImageHeader.image_id) \
             .join(ImageLabel, ImageEmbedding.image_id == ImageLabel.image_id) \
             .filter(ImageEmbedding.common_name.in_(species_names),
                     ImageLabel.label_type == 'user')
            if label_filter != "All":
                query = query.filter(ImageEmbedding.common_name == label_filter)

            for img_id, emb, pred, path, score, true_label in query:
                emb = ast.literal_eval(emb) if isinstance(emb, str) else emb
                all_vecs.append(emb)
                all_labels.append("match" if pred == true_label else "drift")
                all_names.append(pred)
                all_paths.append(path)
                all_ids.append(img_id)
                all_true.append(true_label)
                all_scores.append(score)

        reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, metric="cosine", random_state=42)
        umap_2d = reducer.fit_transform(np.array(all_vecs))

        df_plot = pd.DataFrame({
            "x": umap_2d[:, 0],
            "y": umap_2d[:, 1],
            "type": all_labels,
            "name": all_names,
            "path": all_paths,
            "image_id": all_ids,
            "label_species": all_true,
            "score": all_scores
        })

        canonical_df = df_plot[df_plot["type"] == "canonical species vectors"]
        gold_df = canonical_df[canonical_df["name"].isin(labeled_species)]
        blue_df = canonical_df[~canonical_df["name"].isin(labeled_species)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=blue_df["x"], y=blue_df["y"], mode="markers",
                                 name="canonical species", marker=dict(color="blue", size=10),
                                 hovertext=blue_df["name"]))
        if not gold_df.empty:
            fig.add_trace(go.Scatter(x=gold_df["x"], y=gold_df["y"], mode="markers+text",
                                     name="labeled species (gold)",
                                     marker=dict(color="gold", size=12, line=dict(width=1, color="black")),
                                     text=gold_df["name"], textposition="top center"))

        for label, color in [("match", "green"), ("drift", "red")]:
            df_sub = df_plot[df_plot["type"] == label]
            if not df_sub.empty:
                fig.add_trace(go.Scatter(x=df_sub["x"], y=df_sub["y"], mode="markers",
                                         name=label, marker=dict(color=color, size=10),
                                         hovertext=df_sub["name"]))

        fig.update_layout(height=700, margin=dict(l=40, r=40, t=40, b=40), hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)

        drift_df = df_plot[df_plot["type"] == "drift"]
        if not drift_df.empty:
            st.markdown("### ❌ Drift Outliers")
            st.dataframe(drift_df[["image_id", "label_species", "name", "score"]]
                         .rename(columns={"name": "predicted_species"}), use_container_width=True)
        else:
            st.markdown("✅ No drift detected among image embeddings.")


def speciesnet(limit=100):
    """Summarize SpeciesNet results with confidence distribution and accuracy"""
    with SessionLocal() as session:
        records = session.query(
            ImageHeader.image_id,
            ImageHeader.species_confidence,
            SpeciesFlattened.common_name,
            ImageLabel.label_value
        ).join(SpeciesFlattened, ImageHeader.species_id == SpeciesFlattened.species_id) \
         .outerjoin(ImageLabel, ImageHeader.image_id == ImageLabel.image_id) \
         .filter(ImageHeader.species_detection_method == "speciesnet") \
         .limit(limit).all()

    data = []
    for image_id, conf, predicted, label in records:
        match = predicted and label and predicted.lower().strip() == label.lower().strip()
        data.append({
            "image_id": image_id,
            "label": label,
            "predicted": predicted,
            "confidence score": round(conf * 100, 2) if conf is not None else None,
            "match": "✅" if match else "❌"
        })

    df = pd.DataFrame(data).sort_values("image_id", ascending=False)
    st.subheader("📄 SpeciesNet Evaluation Results")
    st.dataframe(df)

    st.subheader("🔢 Species Detection Count")
    st.bar_chart(df["predicted"].value_counts())

    st.subheader("📊 Confidence Score Distribution")
    bins = [0, 60, 70, 80, 90, 100]
    labels = ["<60%", "60–69%", "70–79%", "80–89%", "90–100%"]
    df["confidence_group"] = pd.cut(df["confidence score"], bins=bins, labels=labels, include_lowest=True)
    st.bar_chart(df["confidence_group"].value_counts().reindex(labels[::-1]))

    acc = df["match"].value_counts().get("✅", 0) / len(df) if len(df) else 0
    st.markdown(f"### ✅ Accuracy: {acc:.2%}")


def clip_region_comparison(limit=100):
    """Compare a species to others within its ecoregion using CLIP distance"""
    ECOREGION_CODE = "NA0528"
    ECOREGION_NAME = "South Central Rockies forests"

    with SessionLocal() as session:
        species_data = session.execute(text("""
            SELECT common_name, class_name
            FROM wildlife.species_by_region
            WHERE ecoregion_code = :code
            ORDER BY common_name
        """), {"code": ECOREGION_CODE}).fetchall()

    species_names = [s[0] for s in species_data]
    species_classes = {s[0]: s[1] for s in species_data}
    target_species = st.selectbox("Target Species", species_names)

    if st.button("Run Regional Comparison"):
        with SessionLocal() as session:
            vecs = session.query(ImageEmbedding.image_embedding) \
                .join(ImageHeader, ImageEmbedding.image_id == ImageHeader.image_id) \
                .join(ImageLabel, ImageHeader.image_id == ImageLabel.image_id) \
                .filter(ImageLabel.label_value == target_species,
                        ImageLabel.label_type == 'user').all()

            if not vecs:
                st.warning(f"No user-labeled embeddings found for '{target_species}'")
                st.stop()

            base_vec = np.mean([np.array(v) for (v,) in vecs], axis=0)

            species_embs = session.execute(text("""
                SELECT common_name, image_embedding
                FROM wildlife.species_embedding
                WHERE common_name = ANY(:species)
            """), {"species": species_names}).fetchall()

            distances = []
            for name, emb in species_embs:
                if name == target_species:
                    continue
                emb = ast.literal_eval(emb) if isinstance(emb, str) else emb
                dist = 1 - np.dot(base_vec, emb) / (np.linalg.norm(base_vec) * np.linalg.norm(emb))
                distances.append((name, dist, species_classes.get(name, "Unknown")))

        if distances:
            df = pd.DataFrame(distances, columns=["Species", "Cosine Distance", "Class"]).sort_values("Cosine Distance")

            class_colors = {"Mammalia": "green", "Aves": "blue", "Reptilia": "orange",
                            "Amphibia": "purple", "Insecta": "gray"}

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = df["Class"].map(class_colors).fillna("black")
            ax.scatter(df["Species"], df["Cosine Distance"], c=colors, alpha=0.8)

            ax.set_xticks([])  # Hide x-tick clutter
            ax.set_ylabel(f"Cosine Distance to '{target_species}'")
            ax.set_title(f"CLIP Similarity: '{target_species}' vs. Other Species in {ECOREGION_NAME}")

            st.pyplot(fig)
            st.markdown("### 🐾 Closest Species")
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.warning("No species embeddings found for this region.")
