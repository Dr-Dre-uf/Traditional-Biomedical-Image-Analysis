import streamlit as st
import numpy as np
from skimage.morphology import closing, disk
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

# ------------------------------------------------------------
# PAGE SETUP & SAFETY
# ------------------------------------------------------------
st.set_page_config(page_title="Medical Imaging Tools", layout="wide")

# Safety Notice (Notebook Requirement: Rigor and Reproducibility)
st.sidebar.warning("⚠️ **Data Privacy Notice:** Do not upload images containing Protected Health Information (PHI) or identifiable patient data.")

st.title("Biomedical Image Analysis Toolkit")
st.markdown("""
This tool implements **Microskill 3** techniques: Texture Analysis for pathology detection 
and Morphological Operations for image enhancement.
""")

# ------------------------------------------------------------
# UTILITY: STANDARDIZATION (Notebook Requirement)
# ------------------------------------------------------------
def standardize_image(image):
    """Ensures images are grayscale and normalized to [0, 255] for consistent analysis."""
    img_array = np.array(image.convert("L"))
    # Min-Max Normalization to ensure consistency across different hospitals/scanners
    img_min, img_max = img_array.min(), img_array.max()
    if img_max > img_min:
        standardized = (img_array - img_min) / (img_max - img_min) * 255
    else:
        standardized = img_array
    return standardized.astype(np.uint8)

# ------------------------------------------------------------
# TOOL 1: TEXTURE FEATURE ANALYSIS (GLCM)
# ------------------------------------------------------------
def texture_feature_page():
    st.header("1. Texture Analysis for Disease Detection")
    st.info("ℹ️ **How it works:** This tool calculates GLCM features. Malignant tissues often disrupt regular patterns, leading to higher contrast and lower correlation.")
    
    # Image Loading Logic
    image_choice = st.radio("Select Dataset:", ["Malignant (Sample)", "Benign (Sample)", "Upload New Scan"], horizontal=True)
    
    if image_choice == "Upload New Scan":
        uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if not uploaded: return
        image = Image.open(uploaded)
    else:
        path = "./data/small_slide_BC.png" if "Malignant" in image_choice else "./data/small_slide_noBC.png"
        try:
            image = Image.open(path)
        except FileNotFoundError:
            st.error("Sample data not found. Please upload an image.")
            return

    processed_img = standardize_image(image)

    # GLCM Calculation (Aligning with Notebook Math)
    # 
    glcm = graycomatrix(processed_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    col1, col2 = st.columns(2)
    col1.image(image, caption="Input Image", use_container_width=True)
    
    with col2:
        st.subheader("Statistical Biomarkers")
        st.metric("Contrast", f"{contrast:.2f}", help="Measures local intensity variation. High values suggest heterogeneity (common in tumors).")
        st.metric("Correlation", f"{correlation:.4f}", help="Measures texture regularity. Lower values suggest disrupted tissue structure.")
        
        if contrast > 800:
            st.warning("High contrast detected: This texture profile is statistically similar to disrupted/malignant tissue patterns.")
        else:
            st.success("Low contrast detected: This profile suggests more organized, uniform tissue.")

# ------------------------------------------------------------
# TOOL 2: MORPHOLOGICAL CLOSING
# ------------------------------------------------------------
def morphological_closing_page():
    st.header("2. Morphological Operations")
    # 
    
    st.markdown("""
    **Instructions:** Use this tool to bridge gaps in structures or remove 'speckle' noise from ultrasounds. 
    Adjust the radius to balance detail preservation with noise reduction.
    """)

    radius = st.select_slider("Structuring Element Radius (Disk Size)", options=range(1, 16), value=5, 
                              help="Larger radius fills larger holes but risks merging separate anatomical structures.")

    uploaded_img = st.file_uploader("Upload Ultrasound", type=["png", "jpg", "jpeg"], key="morph")
    
    if uploaded_img:
        img = standardize_image(Image.open(uploaded_img))
        # Operation: Dilation followed by Erosion
        closed_img = closing(img, disk(radius))

        c1, c2 = st.columns(2)
        c1.image(img, caption="Original Ultrasound", use_container_width=True)
        c2.image(closed_img, caption="Processed (Closed) Image", use_container_width=True)

        with st.expander("Clinical Interpretation"):
            st.write("**Benefits:** Fills small dark gaps in bright lesions, making them easier to segment.")
            st.write("**Risks:** May accidentally fill in dark necrotic centers or small vessels (cysts).")

# ------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------
page = st.sidebar.selectbox("Select Tool", ["Texture Analysis", "Morphological Closing"])

if page == "Texture Analysis":
    texture_feature_page()
else:
    morphological_closing_page()
