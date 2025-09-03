import streamlit as st
import numpy as np
from PIL import Image

def compute_texture_features(image, contrast_threshold, correlation_threshold):
    """Calculates contrast and correlation using NumPy, applying thresholds."""
    img_array = np.array(image)
    # Apply thresholding
    thresholded_img = np.where(img_array > contrast_threshold, 255, 0)
    mean = np.mean(thresholded_img)
    std = np.std(thresholded_img)
    contrast = std**2  # A simple measure of contrast
    # Calculate correlation using the covariance matrix
    reshaped_img_array = thresholded_img.reshape(1, -1)
    covariance_matrix = np.cov(reshaped_img_array)
    # Handle the case where np.cov returns a scalar (single row)
    if covariance_matrix.ndim == 0:
        correlation = 0.0  # Or a more appropriate default value
    else:
        correlation = covariance_matrix[0, 1]  # Correlation between pixels

    # Apply correlation threshold to the reshaped image
    thresholded_correlation_img = np.where(reshaped_img_array > correlation_threshold, 1.0, 0.0)
    correlation = np.mean(thresholded_correlation_img)

    return thresholded_img, contrast, correlation

# --- Initialize session state ---
if 'image' not in st.session_state:
    st.session_state.image = Image.open("./data/small_slide_BC.png")  # Default image

# --- Main Application ---
st.title("Texture Feature Analysis")

# --- Sidebar for Image Selection and Upload ---
with st.sidebar:
    st.header("Image Selection")
    image_type = st.radio("Select Image Type:", ("Malignant", "Benign", "Upload Your Own"))
    # --- Load Images ---
    try:
        image_malignant = Image.open("./data/small_slide_BC.png")
        image_benign = Image.open("./data/small_slide_noBC.png")
    except FileNotFoundError as e:
        st.error(f"Error loading images: {e}.  Make sure images are in the './data' directory.")
        st.stop()
    # --- Image Selection Logic ---
    if image_type == "Malignant":
        st.session_state.image = image_malignant
    elif image_type == "Benign":
        st.session_state.image = image_benign
    else:  # Upload Your Own
        uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg"])
        if uploaded_file is not None:
            st.session_state.image = Image.open(uploaded_file)

# --- Main Panel for Sliders and Results ---
st.header("Feature Calculation")

# --- Sliders for Thresholds ---
contrast_threshold = st.slider("Contrast Threshold", 0.0, 255.0, 128.0, 10.0)  #Adjusted range
correlation_threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.7, 0.01)

# --- Calculate Results ---
thresholded_img, contrast, correlation = compute_texture_features(
    st.session_state.image,
    contrast_threshold,
    correlation_threshold
)

# --- Display Images side by side ---
col1, col2 = st.columns(2)
with col1:
    st.image(st.session_state.image, caption="Original Image", width=300)
with col2:
    st.image(thresholded_img, caption="Thresholded Image", width=300)