import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI đọc ảnh X-ray container", layout="wide")

st.title("AI phân tích ảnh X-ray container")

uploaded = st.file_uploader("Tải ảnh X-ray", type=["jpg", "jpeg", "png", "bmp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            count += 1

            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(
                result,
                "NGHI VAN",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    st.subheader("Ảnh gốc")
    st.image(img, use_container_width=True)

    st.subheader("Ảnh phát hiện nghi vấn")
    st.image(result, use_container_width=True)

    st.success(f"Phát hiện {count} vùng nghi vấn")