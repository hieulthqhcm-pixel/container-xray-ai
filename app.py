import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.set_page_config(
    page_title="X-Ray Cargo Analyzer V5",
    layout="wide"
)

st.title("📦 AI X-Ray Cargo Analyzer V5 Professional")

manifest = st.text_area(
    "📄 Nhập Manifest / Khai báo hàng hóa",
    height=120
)

uploaded = st.file_uploader(
    "📤 Upload ảnh X-ray",
    type=["jpg", "jpeg", "png"]
)

# =========================
# KEYWORDS
# =========================

dense_keywords = [
    "steel",
    "metal",
    "machinery",
    "engine",
    "motor",
    "forklift",
    "iron",
    "equipment",
    "bearing",
    "pipe",
    "machine",
]

light_keywords = [
    "flowerpot",
    "ceramic",
    "plastic",
    "toy",
    "paper",
    "textile",
    "garment",
    "shoe",
    "bag",
    "foam",
]

# =========================
# IMAGE ANALYSIS
# =========================

def analyze_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    brightness = np.mean(gray)

    edges = cv2.Canny(gray, 80, 180)
    edge_density = np.sum(edges > 0) / edges.size

    variance = np.var(gray)

    # Dense detection
    dense_ratio = np.sum(gray < 70) / gray.size

    # Heatmap
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Suspicious zones
    thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)[1]

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        area = cv2.contourArea(c)

        if area > 5000:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))

    return {
        "brightness": brightness,
        "edge_density": edge_density,
        "variance": variance,
        "dense_ratio": dense_ratio,
        "heat": heat,
        "boxes": boxes
    }

# =========================
# MANIFEST ANALYSIS
# =========================

def manifest_expected_density(text):

    text = text.lower()

    dense_score = 0
    light_score = 0

    for k in dense_keywords:
        if k in text:
            dense_score += 1

    for k in light_keywords:
        if k in text:
            light_score += 1

    if dense_score > light_score:
        return "dense"

    return "light"

# =========================
# RISK ENGINE
# =========================

def calculate_risk(result, expected):

    dense_ratio = result["dense_ratio"]

    reasons = []

    score = 0

    # CASE 1
    if expected == "light":

        if dense_ratio > 0.45:
            score += 70
            reasons.append(
                "Ảnh có mật độ hấp thụ tia X cao bất thường so với manifest khai báo hàng nhẹ"
            )

        elif dense_ratio > 0.30:
            score += 40
            reasons.append(
                "Xuất hiện vùng hấp thụ tia X đậm đáng chú ý"
            )

    # CASE 2
    else:

        if dense_ratio < 0.15:
            score += 60
            reasons.append(
                "Manifest khai máy móc/kim loại nhưng ảnh X-ray quá rỗng"
            )

    # Edge density
    if result["edge_density"] > 0.12:
        score += 15
        reasons.append(
            "Mật độ cấu trúc vật thể cao"
        )

    # Variance
    if result["variance"] > 5000:
        score += 15
        reasons.append(
            "Độ biến thiên ảnh lớn"
        )

    score = min(score, 100)

    if score >= 70:
        level = "🔴 RỦI RO CAO"

    elif score >= 40:
        level = "🟠 RỦI RO TRUNG BÌNH"

    else:
        level = "🟢 ÍT NGHI VẤN"

    return score, level, reasons

# =========================
# MAIN
# =========================

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    img = np.array(image)

    st.image(
        img,
        caption="Ảnh X-ray",
        width='stretch'
    )

    result = analyze_image(img)

    expected = manifest_expected_density(manifest)

    score, level, reasons = calculate_risk(
        result,
        expected
    )

    # Draw suspicious boxes ONLY if risk high
    boxed = img.copy()

    if score >= 40:

        for (x, y, w, h) in result["boxes"]:

            cv2.rectangle(
                boxed,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                3
            )

    st.subheader("🔥 Heatmap phân tích")

    st.image(
        result["heat"],
        width='stretch'
    )

    st.subheader("📦 Khu vực nghi vấn")

    st.image(
        boxed,
        width='stretch'
    )

    st.subheader("📊 Chỉ số kỹ thuật")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Độ sáng",
        round(result["brightness"], 2)
    )

    c2.metric(
        "Mật độ cạnh",
        round(result["edge_density"], 3)
    )

    c3.metric(
        "Biến thiên",
        round(result["variance"], 2)
    )

    c4.metric(
        "Tỷ lệ vùng đậm",
        round(result["dense_ratio"], 3)
    )

    st.subheader("🚨 Đánh giá rủi ro")

    st.markdown(f"## {level}")

    st.progress(score / 100)

    st.write(f"Điểm nghi vấn: {score}/100")

    st.subheader("📝 Giải thích")

    if reasons:

        for r in reasons:
            st.write("- " + r)

    else:
        st.write("Không phát hiện dấu hiệu bất thường lớn.")

    st.subheader("📄 Manifest phân loại")

    st.write(f"AI đánh giá manifest thuộc nhóm: **{expected.upper()}**")