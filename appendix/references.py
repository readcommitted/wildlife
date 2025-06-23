"""
references.py — System References & Citations
----------------------------------------------

This Streamlit module provides proper attribution for third-party tools,
datasets, and research used in the Wildlife Vision System.

Includes:
- Citation for SpeciesNet (wildlife species identification model)
- Citation for ExifTool (image metadata extraction)
- Citation for WWF WildFinder (species occurrence data)

Maintaining accurate references ensures scientific credibility,
respect for open-source projects, and reproducibility for others
interested in the system's technical foundations.

Dependencies:
- Streamlit for display

Author: Matt Scardino
Project: Wildlife Vision System
"""

import streamlit as st

# --- SpeciesNet Citation ---
st.write("### Citing SpeciesNet")
st.write("For species identification, we use SpeciesNet:")
st.code(
"""@article{gadot2024crop,
  title={To crop or not to crop: Comparing whole-image and cropped classification on a large dataset of camera trap images},
  author={Gadot, Tomer and Istrate, Ștefan and Kim, Hyungwon and Morris, Dan and Beery, Sara and Birch, Tanya and Ahumada, Jorge},
  journal={IET Computer Vision},
  year={2024},
  publisher={Wiley Online Library}
}"""
)

# --- ExifTool Citation ---
st.write("### Citing ExifTool")
st.write("For image metadata processing, we use ExifTool:")
st.code("- ExifTool by Phil Harvey: https://exiftool.org")

# --- WWF WildFinder Citation ---
st.write("### Citing World Wildlife Fund")
st.write("World Wildlife Fund. (n.d.). WildFinder database. World Wildlife Fund.")
st.write("https://www.worldwildlife.org/publications/wildfinder-database")
