# Wildlife Image Processing & Semantic Search System

The **Wildlife Image Processing & Semantic Search System** is a prototype platform for organizing, identifying, and exploring wildlife imagery using modern AI models, geospatial analysis, and semantic search.

This project combines computer vision, location data, and embeddings to assist wildlife enthusiasts, researchers, and photographers in documenting species, discovering patterns, and building richer digital field notes.

---

## Current Features

- ✅ Species identification using CLIP-based embeddings  
- ✅ Regional species filtering based on ecoregion boundaries  
- ✅ Smart similarity search with text prompts  
- ✅ Location-aware ranking to prioritize species likely present  
- ✅ Image metadata extraction and geolocation  
- ✅ Interactive Streamlit UI for image ingestion, validation, and analysis  
- ✅ UMAP and embedding space visualization tools  
- ✅ Early-stage support for SpeciesNet wildlife classification  

---

## Project Focus

This system is an evolving foundation designed to test core functionality and explore how AI can assist with wildlife documentation and discovery. The current version emphasizes practical workflows and proof-of-concept features, with the architecture built for future expansion.

---

## Technical Overview

- **Streamlit UI** for interactive tools  
- **PostgreSQL + pgvector** for storage and similarity search  
- **OpenAI CLIP models** for generating image and text embeddings  
- **Species presence filtering** based on known ecological regions  
- **Location metadata enrichment** via OpenStreetMap  

---

## Future Enhancements

- Broader species detection models and accuracy improvements  
- Automated habitat-aware species ranking  
- Expanded geospatial tools (parks, sub-regions, polygons)  
- Enhanced search capabilities across image, text, and location fields  
- Data quality validation and drift detection  

---

## Notes

This project is under active development, with core workflows prioritized for testing and iteration. Contributions and feedback are welcome as the system evolves toward a more complete solution for wildlife-focused image analysis.

## References and Acknowledgments

This project builds upon key models and datasets:

- [SpeciesNet](https://github.com/visionforwildlife/speciesnet) for supervised species classification  
- [OpenAI CLIP](https://openai.com/research/clip) for semantic image and text embeddings  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection and cropping  
- [World Wildlife Fund WildFinder Database](https://www.worldwildlife.org/publications/wildfinder-database) for ecoregion and species range filtering  
- [ExifTool](https://exiftool.org) for metadata extraction  

For additional technical details, see the [final report](docs/MScardino_Data_Science_Practicum_Final.pdf).
