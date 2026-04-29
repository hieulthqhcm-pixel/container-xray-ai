import streamlit as st
import numpy as np
from PIL import Image
import cv2

# =====================================
# PAGE
# =====================================

st.set_page_config(
    page_title="X-Ray Cargo Analyzer V6",
    layout="wide"
)

st.title("📦 X-Ray Cargo Analyzer V6 PROFESSIONAL")

manifest = st.text_area(
    "📄 Manifest / Khai báo hàng hóa",
    height=120
)

uploaded = st.file_uploader(
    "📤 Upload ảnh X-ray",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# MANIFEST AI ENGINE
# =====================================

dense_keywords = [
    "steel",
    "iron",
    "metal",
    "machinery",
    "machine",
    "engine",
    "motor",
    "forklift",
    "equipment",
    "pipe",
    "bearing",
    "compressor",
    "generator",
    "pump",
]

medium_keywords = [
    "ceramic",
    "tile",
    "brick",
    "stone",
    "cement",
    "chemical",
    "powder",
    "fertilizer",
]

light_keywords = [
    "plastic",
    "paper",
    "textile",
    "garment",
    "toy",
    "shoe",
    "bag",
    "foam",
    "cotton",
]

def analyze_manifest(text):

    text = text.lower()

    dense = 0
    medium = 0
    light = 0

    for k in dense_keywords:
        if k in text:
            dense += 1

    for k in medium_keywords:
        if k in text:
            medium += 1

    for k in light_keywords:
        if k in text:
            light += 1

    if dense >= medium and dense >= light:
        return "dense"

    if medium >= light:
        return "medium"

    return "light"

# =====================================
# IMAGE ANALYSIS
# =====================================

def image_analysis(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    brightness = np.mean(gray)

    variance = np.var(gray)

    edges = cv2.Canny(gray, 70, 180)

    edge_density = np.sum(edges > 0) / edges.size

    # density zones
    dark_ratio = np.sum(gray < 70) / gray.size
    medium_ratio = np.sum((gray >= 70) & (gray < 140)) / gray.size
    bright_ratio = np.sum(gray >= 140) / gray.size

    # texture
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    lap = cv2.Laplacian(blur, cv2.CV_64F)

    texture_score = np.var(lap)

    # suspicious regions
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    suspicious = []

    h_img, w_img = gray.shape

    for c in contours:

        area = cv2.contourArea(c)

        if area < 7000:
            continue

        x, y, w, h = cv2.boundingRect(c)

        aspect = w / (h + 1)

        region = gray[y:y+h, x:x+w]

        region_dark = np.sum(region < 70) / region.size

        # intelligent filtering
        if region_dark > 0.45:

            suspicious.append({
                "box": (x, y, w, h),
                "dark": region_dark,
                "aspect": aspect,
                "area": area
            })

    # heatmap
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    return {
        "gray": gray,
        "brightness": brightness,
        "variance": variance,
        "edge_density": edge_density,
        "dark_ratio": dark_ratio,
        "medium_ratio": medium_ratio,
        "bright_ratio": bright_ratio,
        "texture_score": texture_score,
        "suspicious": suspicious,
        "heat": heat
    }

# =====================================
# AI RISK ENGINE
# =====================================

def risk_engine(manifest_type, result):

    score = 0

    reasons = []

    dark = result["dark_ratio"]

    texture = result["texture_score"]

    edge = result["edge_density"]

    suspicious_count = len(result["suspicious"])

    # =========================
    # LIGHT CARGO
    # =========================

    if manifest_type == "light":

        if dark > 0.40:
            score += 50
            reasons.append(
                "Ảnh có mật độ hấp thụ tia X cao bất thường đối với hàng nhẹ"
            )

        if suspicious_count >= 2:
            score += 25
            reasons.append(
                "Xuất hiện nhiều vùng hấp thụ tia X đậm"
            )

    # =========================
    # MEDIUM
    # =========================

    elif manifest_type == "medium":

        if dark > 0.60:
            score += 40
            reasons.append(
                "Mật độ ảnh vượt ngưỡng thông thường của hàng trung bình"
            )

        if suspicious_count >= 3:
            score += 20
            reasons.append(
                "Nhiều vùng hấp thụ bất thường"
            )

    # =========================
    # DENSE CARGO
    # =========================

    else:

        if dark < 0.12:
            score += 60
            reasons.append(
                "Manifest khai máy móc/kim loại nhưng ảnh quá rỗng"
            )

        if edge < 0.03:
            score += 20
            reasons.append(
                "Thiếu cấu trúc vật thể tương ứng hàng máy móc"
            )

    # texture
    if texture > 1500:
        score += 15
        reasons.append(
            "Texture ảnh phức tạp"
        )

    # variance
    if result["variance"] > 5500:
        score += 15
        reasons.append(
            "Biến thiên ảnh cao"
        )

    score = min(score, 100)

    # =========================
    # LEVEL
    # =========================

    if score >= 75:
        level = "🔴 RỦI RO CAO"

    elif score >= 45:
        level = "🟠 RỦI RO TRUNG BÌNH"

    else:
        level = "🟢 ÍT NGHI VẤN"

    return score, level, reasons

# =====================================
# DRAW SMART BOXES
# =====================================

def draw_boxes(img, suspicious):

    out = img.copy()

    for s in suspicious:

        x, y, w, h = s["box"]

        cv2.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            4
        )

        cv2.putText(
            out,
            "Suspicious",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )

    return out

# =====================================
# MAIN
# =====================================

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    img = np.array(image)

    st.subheader("🖼 Ảnh X-ray")

    st.image(
        img,
        width='stretch'
    )

    # analyze
    manifest_type = analyze_manifest(manifest)

    result = image_analysis(img)

    score, level, reasons = risk_engine(
        manifest_type,
        result
    )

    # draw only real suspicious
    boxed = draw_boxes(
        img,
        result["suspicious"]
    )

    # =====================================
    # DISPLAY
    # =====================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🔥 Heatmap")

        st.image(
            result["heat"],
            width='stretch'
        )

    with col2:

        st.subheader("📦 Vùng nghi vấn")

        st.image(
            boxed,
            width='stretch'
        )

    # =====================================
    # METRICS
    # =====================================

    st.subheader("📊 Phân tích kỹ thuật")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Dark Ratio",
        round(result["dark_ratio"], 3)
    )

    c2.metric(
        "Edge Density",
        round(result["edge_density"], 3)
    )

    c3.metric(
        "Texture",
        round(result["texture_score"], 1)
    )

    c4.metric(
        "Variance",
        round(result["variance"], 1)
    )

    c5.metric(
        "Suspicious Zones",
        len(result["suspicious"])
    )

    # =====================================
    # RISK
    # =====================================

    st.subheader("🚨 Đánh giá rủi ro")

    st.markdown(f"## {level}")

    st.progress(score / 100)

    st.write(f"Điểm nghi vấn: {score}/100")

    # =====================================
    # EXPLANATION
    # =====================================

    st.subheader("📝 Giải thích AI")

    if reasons:

        for r in reasons:
            st.write("- " + r)

    else:
        st.write("Không phát hiện dấu hiệu bất thường lớn.")

    # =====================================
    # MANIFEST CLASSIFICATION
    # =====================================

    st.subheader("📄 AI phân loại Manifest")

    st.write(f"Loại hàng AI đánh giá: **{manifest_type.upper()}**")

    # =====================================
    # PROFESSIONAL SUMMARY
    # =====================================

    st.subheader("📌 Kết luận nghiệp vụ")

    if score >= 75:

        st.error(
            "Khuyến nghị kiểm tra thực tế container do có dấu hiệu sai lệch manifest."
        )

    elif score >= 45:

        st.warning(
            "Khuyến nghị soi chiếu tăng cường và kiểm tra hồ sơ."
        )

    else:

        st.success(
            "Chưa phát hiện dấu hiệu bất thường rõ ràng."
        )