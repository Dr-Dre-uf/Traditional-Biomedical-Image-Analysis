import streamlit as st
import numpy as np
from skimage.morphology import closing, disk
from PIL import Image

# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(page_title="Medical Imaging Tools", layout="wide")
st.title("Medical Imaging Toolkit")

# ------------------------------------------------------------
# PAGE SELECTOR
# ------------------------------------------------------------
page = st.sidebar.selectbox(
    "Select Tool",
    ["Texture Feature Analysis", "Morphological Closing"]
)


# ------------------------------------------------------------
# --- PAGE 1: Texture Feature Analysis ------------------------
# ------------------------------------------------------------
def compute_texture_features(image, contrast_threshold, correlation_threshold):
    """Calculates contrast and correlation using NumPy, applying thresholds."""
    img_array = np.array(image)

    # Contrast thresholding
    thresholded_img = np.where(img_array > contrast_threshold, 255, 0)

    mean = np.mean(thresholded_img)
    std = np.std(thresholded_img)
    contrast = std ** 2  # simple contrast measure

    # Correlation calculation
    reshaped = thresholded_img.reshape(1, -1)
    covariance_matrix = np.cov(reshaped)

    if covariance_matrix.ndim == 0:
        correlation = 0.0
    else:
        correlation = covariance_matrix[0, 0] if covariance_matrix.size == 1 else covariance_matrix[0, 1]

    thresholded_corr = np.where(reshaped > correlation_threshold, 1.0, 0.0)
    correlation = np.mean(thresholded_corr)

    return thresholded_img, contrast, correlation


def texture_feature_page():
    st.header("Texture Feature Analysis")

    # --- Initialize default images
    try:
        image_malignant = Image.open("./data/small_slide_BC.png")
        image_benign = Image.open("./data/small_slide_noBC.png")
    except FileNotFoundError:
        st.error("Could not load default images from ./data/")
        return

    # Sidebar
    st.sidebar.subheader("Texture Image Selection")
    image_type = st.sidebar.radio("Select Image Type:", ("Malignant", "Benign", "Upload Your Own"))

    if image_type == "Malignant":
        image = image_malignant
    elif image_type == "Benign":
        image = image_benign
    else:
        uploaded = st.sidebar.file_uploader("Upload Image", type=["png", "jpg"])
        if uploaded:
            image = Image.open(uploaded)
        else:
            st.info("Upload an image to continue.")
            return

    # Sliders
    contrast_threshold = st.slider("Contrast Threshold", 0.0, 255.0, 128.0, 10.0)
    correlation_threshold = st.slider("Correlation Threshold", 0.0, 1.0, 0.7, 0.01)

    # Compute texture features
    thresholded_img, contrast, correlation = compute_texture_features(
        image, contrast_threshold, correlation_threshold
    )

    # Display images
    col1, col2 = st.columns(2)
    col1.image(image, caption="Original Image", width=300)
    col2.image(thresholded_img, caption="Thresholded Image", width=300)

    # Display metrics
    st.subheader("Results")
    st.write(f"**Contrast:** {contrast:.4f}")
    st.write(f"**Correlation:** {correlation:.4f}")


# ------------------------------------------------------------
# --- PAGE 2: Morphological Closing Tool ---------------------
# ------------------------------------------------------------
def apply_morphological_closing(image, radius=5):
    selem = disk(radius)
    closed_image = closing(image, selem)
    return closed_image


def morphological_closing_page():
    st.header("Microskill 3.2: Morphological Closing in Medical Imaging")
    st.warning("⚠️ Do not upload sensitive or personal data.")

    st.sidebar.subheader("Ultrasound Image Selection")
    use_uploaded = st.sidebar.checkbox("Upload your own image")

    if use_uploaded:
        uploaded_img = st.sidebar.file_uploader("Upload Ultrasound Image", type=["jpg", "jpeg", "png"])
    else:
        uploaded_img = None
        sample_path = "data/breast_US.png"

    # Morphological radius
    radius = st.sidebar.slider("Structuring Element Radius", 1, 15, 5)

    # Load image (OpenCV removed)
    if use_uploaded and uploaded_img:
        img = np.array(Image.open(uploaded_img).convert("L"))
    elif not use_uploaded:
        try:
            img = np.array(Image.open(sample_path).convert("L"))
        except FileNotFoundError:
            st.error(f"Sample image not found at {sample_path}")
            return
    else:
        img = None

    if img is not None:
        closed_img = apply_morphological_closing(img, radius)

        col1, col2 = st.columns(2)
        col1.image(img, caption="Original Ultrasound", use_container_width=True)
        col2.image(closed_img, caption="Morphologically Closed", use_container_width=True)

        st.markdown("""
        ### Interpretation
        - **Benefit:** Smooths small holes, reduces speckle noise, and makes structures more continuous.
        - **Drawback:** May remove fine details such as microcalcifications or subtle boundaries.
        """)
    else:
        st.info("Please upload an image or use the sample.")


# ------------------------------------------------------------
# PAGE ROUTING
# ------------------------------------------------------------
if page == "Texture Feature Analysis":
    texture_feature_page()
else:
    morphological_closing_page()
