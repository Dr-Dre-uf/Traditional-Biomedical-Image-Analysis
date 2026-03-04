import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

# ------------------------------------------------------------
# PAGE SETUP & SIDEBAR CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Medical Imaging Toolkit", layout="wide")

# Persistent Sidebar Safety Notice
st.sidebar.error("⚠️ **Notice:** Do not upload private/sensitive medical data.")

st.sidebar.title("Navigation & Controls")
page = st.sidebar.radio("Select Tool", ["Texture Feature Analysis", "Morphological Closing"])

# ------------------------------------------------------------
# SHARED UTILITIES
# ------------------------------------------------------------
def standardize_image(image, apply_norm):
    """Normalizes image based on Notebook 3.3 standards."""
    img_array = np.array(image.convert("L"))
    if apply_norm:
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return img_array

# ------------------------------------------------------------
# TOOL 1: TEXTURE FEATURE ANALYSIS
# ------------------------------------------------------------
if page == "Texture Feature Analysis":
    st.title("Texture Feature Analysis")
    
    # Sidebar Controls
    st.sidebar.subheader("Image Settings")
    dataset = st.sidebar.selectbox("Dataset Source", ["Malignant Sample", "Benign Sample", "Custom Upload"])
    apply_standard = st.sidebar.checkbox("Standardize Intensities", value=True, help="Normalizes pixel values to [0, 255] to ensure reproducibility.")
    
    # Distance and Angle for GLCM (Notebook 3.1)
    st.sidebar.subheader("GLCM Parameters")
    dist = st.sidebar.slider("Pixel Distance", 1, 5, 1)
    angle = st.sidebar.selectbox("Angle (Radians)", [0, np.pi/4, np.pi/2, 3*np.pi/4], format_func=lambda x: f"{int(np.degrees(x))}°")

    # Image Loading
    if dataset == "Custom Upload":
        uploaded = st.sidebar.file_uploader("Upload Image", type=["png", "jpg"])
        if not uploaded:
            st.info("Please upload an image in the sidebar.")
            st.stop()
        img_raw = Image.open(uploaded)
    else:
        path = "./data/small_slide_BC.png" if "Malignant" in dataset else "./data/small_slide_noBC.png"
        try:
            img_raw = Image.open(path)
        except:
            st.error("Sample image not found.")
            st.stop()

    img = standardize_image(img_raw, apply_standard)
    
    # Computation
    glcm = graycomatrix(img, distances=[dist], angles=[angle], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Display
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(img, caption=f"Analyzed View ({dataset})", use_container_width=True)
    
    with col2:
        st.subheader("Metrics")
        st.metric("Contrast", f"{contrast:.2f}")
        st.metric("Correlation", f"{correlation:.4f}")
        
        with st.expander("What do these mean?"):
            st.write("**Contrast:** Higher values indicate sharper local variations, often seen in irregular tumor structures.")
            st.write("**Correlation:** Measures linear dependency. Lower values suggest a loss of organized tissue patterns.")

# ------------------------------------------------------------
# TOOL 2: MORPHOLOGICAL CLOSING
# ------------------------------------------------------------
else:
    st.title("Morphological Closing Tool")
    
    # Sidebar Controls
    st.sidebar.subheader("Closing Parameters")
    radius = st.sidebar.slider("Disk Radius", 1, 20, 5, help="Size of the structuring element.")
    show_diff = st.sidebar.checkbox("Show Difference Map", value=True, help="Highlights exactly what was filled or changed.")
    
    uploaded_us = st.sidebar.file_uploader("Upload Ultrasound Scan", type=["png", "jpg", "jpeg"])

    if not uploaded_us:
        st.info("Upload an ultrasound scan in the sidebar to begin.")
    else:
        img = standardize_image(Image.open(uploaded_us), True)
        closed_img = closing(img, disk(radius))
        
        # Display Results
        c1, c2 = st.columns(2)
        c1.image(img, caption="Original Scan", use_container_width=True)
        c2.image(closed_img, caption=f"Closed (Radius {radius})", use_container_width=True)

        if show_diff:
            st.subheader("Changes Applied")
            # Calculate the difference to show what 'Closing' actually did
            diff = closed_img.astype(np.float32) - img.astype(np.float32)
            fig, ax = plt.subplots()
            im = ax.imshow(diff, cmap='magma')
            plt.colorbar(im)
            ax.axis('off')
            st.pyplot(fig)
            st.caption("Bright spots show areas where dark holes or gaps were filled.")

        st.markdown("""
        ### Clinical Trade-offs
        * **Benefit:** Connects fragmented structures (e.g., vessel walls).
        * **Risk:** A radius that is too large will merge separate tissues or erase small, dark pathological features like cysts.
        """)
