import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import io

sns.set_style("whitegrid")

st.title("Image Processing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_gray = image.convert("L")
    img_np_gray = np.array(image_gray)
    img_np_color = np.array(image.convert("RGB"))

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # LOG TRANSFORMATION
    st.subheader("Log Transformation")
    c = 255 / np.log(1 + np.max(img_np_gray))
    log_image = c * np.log(1 + img_np_gray)
    log_image = np.array(log_image, dtype=np.uint8)
    st.image(log_image, caption="Log Transformed Image", use_column_width=True, clamp=True)

    # HISTOGRAM AND EQUALIZATION
    st.subheader("Histogram Equalization")
    equalized_img = cv2.equalizeHist(img_np_gray)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].hist(img_np_gray.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    axs[0].set_title("Original Histogram")
    axs[1].hist(equalized_img.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
    axs[1].set_title("Equalized Histogram")
    st.pyplot(fig)
    st.image(equalized_img, caption="Equalized Image", use_column_width=True, clamp=True)

    # IMAGE ROTATION
    st.subheader("Image Rotation")
    angle = st.slider("Rotation Angle", min_value=-180, max_value=180, value=60)
    rotated_image = image.rotate(angle, expand=True)
    st.image(rotated_image, caption=f"Rotated Image ({angle}°)", use_column_width=True)

    # SHEARING
    st.subheader("Shearing")
    shear_factor = st.slider("Shear Factor", min_value=0.0, max_value=1.0, value=0.3)
    rows, cols, _ = img_np_color.shape
    shear_matrix_horizontal = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    shear_matrix_vertical = np.float32([[1, 0, 0], [shear_factor, 1, 0]])
    new_cols = int(cols + shear_factor * rows)
    new_rows = int(rows + shear_factor * cols)
    sheared_h = cv2.warpAffine(img_np_color, shear_matrix_horizontal, (new_cols, rows))
    sheared_v = cv2.warpAffine(img_np_color, shear_matrix_vertical, (cols, new_rows))
    st.image(sheared_h, caption="Sheared Horizontally", use_column_width=True, clamp=True)
    st.image(sheared_v, caption="Sheared Vertically", use_column_width=True, clamp=True)

    # SHARPENING
    st.subheader("Sharpening (Laplacian Filter)")
    laplacian = cv2.Laplacian(img_np_gray, ddepth=cv2.CV_64F)
    laplacian_uint8 = cv2.convertScaleAbs(laplacian)
    sharpened = cv2.addWeighted(img_np_gray, 1.0, laplacian_uint8, 1.0, 0)
    st.image(sharpened, caption="Sharpened Image", use_column_width=True, clamp=True)

    # GAUSSIAN NOISE BLURRING
    st.subheader("Gaussian Noise Blurring")
    blurred = cv2.GaussianBlur(img_np_color, ksize=(25, 25), sigmaX=0)  
    st.image(blurred, caption="Gaussian Blurred Image", use_column_width=True, clamp=True)

    # CANNY EDGE DETECTION
    st.subheader("Canny Edge Detection")
    blurred_for_canny = cv2.GaussianBlur(img_np_gray, (5, 5), 1.4)
    st.image(blurred_for_canny, caption="Step 1: Gaussian Smoothing", use_column_width=True, clamp=True)

    sobelx = cv2.Sobel(blurred_for_canny, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred_for_canny, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude_uint8 = cv2.convertScaleAbs(magnitude)
    st.image(magnitude_uint8, caption="Step 2: Gradient Magnitude (Sobel)", use_column_width=True, clamp=True)

    direction = np.arctan2(sobely, sobelx) * 180 / np.pi
    direction[direction < 0] += 180

    edges = cv2.Canny(blurred_for_canny, 100, 200)
    st.image(edges, caption="Step 3: Final Canny Edge Detection Output", use_column_width=True, clamp=True) 
