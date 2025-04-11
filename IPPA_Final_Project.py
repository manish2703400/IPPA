import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import skimage as ski

st.title("Image Processing Web App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale
    image_gray = image.convert("L")
    img_np = np.array(image_gray)

    # ------------------- Log Transformation -------------------
    c = 255 / np.log(1 + np.max(img_np))
    log_image = c * np.log(1 + img_np)
    log_image = np.array(log_image, dtype=np.uint8)

    st.subheader("Log Transformation")
    col1, col2 = st.columns(2)
    col1.image(img_np, caption="Original Grayscale", use_column_width=True, channels="GRAY")
    col2.image(log_image, caption="Log Transformed", use_column_width=True, channels="GRAY")

    # ------------------- Histogram Equalization -------------------
    equalized_img = cv2.equalizeHist(img_np)

    st.subheader("Histogram Equalization")
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax[0, 0].imshow(img_np, cmap='gray')
    ax[0, 0].set_title("Original")
    ax[0, 0].axis('off')
    ax[0, 1].hist(img_np.ravel(), bins=256, color='blue', alpha=0.7)
    ax[0, 1].set_title("Original Histogram")
    ax[1, 0].imshow(equalized_img, cmap='gray')
    ax[1, 0].set_title("Equalized")
    ax[1, 0].axis('off')
    ax[1, 1].hist(equalized_img.ravel(), bins=256, color='green', alpha=0.7)
    ax[1, 1].set_title("Equalized Histogram")
    st.pyplot(fig)

    # ------------------- Image Rotation -------------------
    angle = st.slider("Select rotation angle", 0, 360, 60)
    rotated_image = image.rotate(angle, expand=False)

    st.subheader("Image Rotation")
    col1, col2 = st.columns(2)
    col1.image(image, caption="Original", use_column_width=True)
    col2.image(rotated_image, caption=f"Rotated {angle}°", use_column_width=True)

    # ------------------- Shearing -------------------
    img_color = image.convert("RGB")
    img_np_1 = np.array(img_color)
    rows, cols, ch = img_np_1.shape
    shear_factor = 0.3
    shear_matrix_horizontal = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    shear_matrix_vertical = np.float32([[1, 0, 0], [shear_factor, 1, 0]])
    new_cols = int(cols + shear_factor * rows)
    new_rows = int(rows + shear_factor * cols)
    sheared_img_horizontal = cv2.warpAffine(img_np_1, shear_matrix_horizontal, (new_cols, rows))
    sheared_img_vertical = cv2.warpAffine(img_np_1, shear_matrix_vertical, (cols, new_rows))

    st.subheader("Image Shearing")
    st.image([image, sheared_img_horizontal, sheared_img_vertical], caption=["Original", "Sheared Horizontal", "Sheared Vertical"], use_column_width=True)

    # ------------------- Sharpening using Laplacian Filter -------------------
    laplacian = cv2.Laplacian(img_np, ddepth=cv2.CV_64F)
    laplacian_uint8 = cv2.convertScaleAbs(laplacian)
    sharpened = cv2.addWeighted(img_np, 1.0, laplacian_uint8, 1.0, 0)

    st.subheader("Laplacian Sharpening")
    st.image([img_np, laplacian_uint8, sharpened], caption=["Original", "Laplacian Filter", "Sharpened"], use_column_width=True)

    # ------------------- Gaussian Noise Blurring -------------------
    blurred_img_X = cv2.GaussianBlur(img_np_1, ksize=(25, 25), sigmaX=0)
    blurred_img_Y = cv2.GaussianBlur(img_np_1, ksize=(25, 25), sigmaX=0, sigmaY=0)

    st.subheader("Gaussian Noise Blurring")
    st.image([img_np_1, blurred_img_X, blurred_img_Y], caption=["Original", "Blurred X", "Blurred Y"], use_column_width=True)
else:
    st.warning("Please upload an image to proceed.")
