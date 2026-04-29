import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.set_page_config(
    page_title="X-Ray Cargo Analyzer V6.5",
    layout="wide"
)

st.title("📦 X-Ray Cargo Analyzer V6.5 SMART CARGO REGION")

manifest = st.text_area(
    "📄 Manifest / Khai báo hàng hóa",
    height=120
)

uploaded = st.file_uploader(
    "📤 Upload ảnh X-ray",
    type=["jpg", "jpeg", "png"]
)

dense_keywords = [
    "steel", "iron", "metal", "machinery", "machine",
    "engine", "motor", "forklift", "equipment", "pipe",
    "bearing", "compressor", "generator", "pump"
]

medium_keywords = [
    "ceramic", "tile", "brick", "stone", "cement",
    "chemical", "powder", "fertilizer", "flowerpot",
    "gạch", "gach", "gốm", "gom"
]

light_keywords = [
    "plastic", "paper", "textile", "garment", "toy",
    "shoe", "bag", "foam", "cotton", "carton", "cartons"
]

def analyze_manifest(text):
    text = text.lower()

    dense = sum(1 for k in dense_keywords if k in text)
    medium = sum(1 for k in medium_keywords if k in text)
    light = sum(1 for k in light_keywords if k in text)

    if dense >= medium and dense >= light and dense > 0:
        return "dense"

    if medium >= light and medium > 0:
        return "medium"

    if light > 0:
        return "light"

    return "unknown"

def crop_cargo_region(gray):
    mask = gray < 235

    coords = np.column_stack(np.where(mask))

    if len(coords) == 0:
        return gray, (0, 0, gray.shape[1], gray.shape[0])

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    pad = 20

    y0 = max(y0 - pad, 0)
    x0 = max(x0 - pad, 0)
    y1 = min(y1 + pad, gray.shape[0])
    x1 = min(x1 + pad, gray.shape[1])

    crop = gray[y0:y1, x0:x1]

    return crop, (x0, y0, x1, y1)

def image_analysis(img):
    gray_full = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray, crop_box = crop_cargo_region(gray_full)

    brightness = np.mean(gray)
    variance = np.var(gray)

    edges = cv2.Canny(gray, 60, 160)
    edge_density = np.sum(edges > 0) / edges.size

    dark_ratio = np.sum(gray < 85) / gray.size
    very_dark_ratio = np.sum(gray < 55) / gray.size
    medium_ratio = np.sum((gray >= 85) & (gray < 170)) / gray.size
    bright_ratio = np.sum(gray >= 170) / gray.size

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(blur, cv2.CV_64F)
    texture_score = np.var(lap)

    thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    suspicious = []

    h_img, w_img = gray.shape

    for c in contours:
        area = cv2.contourArea(c)

        if area < 1500:
            continue

        x, y, w, h = cv2.boundingRect(c)

        if x < 5 or y < 5 or x + w > w_img - 5 or y + h > h_img - 5:
            continue

        aspect = w / (h + 1)
        region = gray[y:y+h, x:x+w]
        region_dark = np.sum(region < 85) / region.size

        if aspect > 10 or aspect < 0.08:
            continue

        if region_dark > 0.22:
            suspicious.append({
                "box": (x, y, w, h),
                "dark": region_dark,
                "aspect": aspect,
                "area": area
            })

    heat_src = cv2.resize(gray, (800, 500))
    heat = cv2.applyColorMap(heat_src, cv2.COLORMAP_JET)

    enhanced = cv2.equalizeHist(gray)
    enhanced = cv2.resize(enhanced, (800, 500))

    return {
        "gray": gray,
        "crop_box": crop_box,
        "brightness": brightness,
        "variance": variance,
        "edge_density": edge_density,
        "dark_ratio": dark_ratio,
        "very_dark_ratio": very_dark_ratio,
        "medium_ratio": medium_ratio,
        "bright_ratio": bright_ratio,
        "texture_score": texture_score,
        "suspicious": suspicious,
        "heat": heat,
        "enhanced": enhanced
    }

def risk_engine(manifest_type, result):
    score = 0
    reasons = []

    dark = result["dark_ratio"]
    very_dark = result["very_dark_ratio"]
    texture = result["texture_score"]
    edge = result["edge_density"]
    variance = result["variance"]
    suspicious_count = len(result["suspicious"])

    machinery_like = (
        edge > 0.045
        or texture > 450
        or suspicious_count >= 2
        or very_dark > 0.025
    )

    uniform_like = (
        edge < 0.035
        and texture < 350
        and suspicious_count <= 1
        and dark < 0.22
    )

    if manifest_type == "light":
        if dark > 0.28:
            score += 50
            reasons.append("Manifest khai hàng nhẹ nhưng vùng hàng có mật độ X-ray cao.")
        if machinery_like:
            score += 30
            reasons.append("Ảnh có dấu hiệu cấu trúc phức tạp không phù hợp hàng nhẹ.")

    elif manifest_type == "medium":
        if dark > 0.42:
            score += 35
            reasons.append("Manifest khai hàng trung bình nhưng mật độ vùng hàng khá cao.")
        if machinery_like and not uniform_like:
            score += 35
            reasons.append("Ảnh có cấu trúc phức tạp, cần đối chiếu với hàng khai báo.")
        if suspicious_count >= 3:
            score += 20
            reasons.append("Có nhiều vùng hấp thụ tia X đáng chú ý.")

    elif manifest_type == "dense":
        if dark < 0.12:
            score += 45
            reasons.append("Manifest khai hàng đặc/kim loại nhưng ảnh vùng hàng khá rỗng.")
        if edge < 0.025:
            score += 20
            reasons.append("Thiếu cấu trúc tương ứng hàng máy móc/kim loại.")

    else:
        if machinery_like:
            score += 45
            reasons.append("Manifest chưa rõ nhóm hàng nhưng ảnh có cấu trúc phức tạp.")
        else:
            score += 20
            reasons.append("Manifest chưa đủ rõ để đối chiếu chắc chắn.")

    if variance > 3500:
        score += 10
        reasons.append("Độ biến thiên ảnh cao.")

    if texture > 700:
        score += 10
        reasons.append("Texture ảnh phức tạp.")

    score = min(score, 100)

    if score >= 75:
        level = "🔴 RỦI RO CAO"
    elif score >= 45:
        level = "🟠 RỦI RO TRUNG BÌNH"
    else:
        level = "🟢 ÍT NGHI VẤN"

    if not reasons:
        reasons.append("Không phát hiện dấu hiệu bất thường lớn theo chỉ số hiện tại.")

    return score, level, reasons

def draw_boxes_on_crop(gray_crop, suspicious):
    out = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)

    for s in suspicious:
        x, y, w, h = s["box"]

        cv2.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

        cv2.putText(
            out,
            "VUNG DAM",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    return out

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    st.subheader("🖼 Ảnh X-ray gốc")
    st.image(img, width="stretch")

    manifest_type = analyze_manifest(manifest)
    result = image_analysis(img)

    score, level, reasons = risk_engine(manifest_type, result)

    boxed = draw_boxes_on_crop(
        result["gray"],
        result["suspicious"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Heatmap vùng hàng")
        st.image(result["heat"], width="stretch")

    with col2:
        st.subheader("📦 Vùng hàng / vùng đậm")
        st.image(boxed, width="stretch")

    st.subheader("🔍 Ảnh tăng tương phản vùng hàng")
    st.image(result["enhanced"], width="stretch", clamp=True)

    st.subheader("📊 Phân tích kỹ thuật vùng hàng")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Dark Ratio", round(result["dark_ratio"], 3))
    c2.metric("Very Dark", round(result["very_dark_ratio"], 3))
    c3.metric("Edge Density", round(result["edge_density"], 3))
    c4.metric("Texture", round(result["texture_score"], 1))
    c5.metric("Suspicious Zones", len(result["suspicious"]))

    st.subheader("🚨 Đánh giá rủi ro")

    st.markdown(f"## {level}")
    st.progress(score / 100)
    st.write(f"Điểm nghi vấn: {score}/100")

    st.subheader("📝 Giải thích AI")
    for r in reasons:
        st.write("- " + r)

    st.subheader("📄 AI phân loại Manifest")
    st.write(f"Loại hàng AI đánh giá: **{manifest_type.upper()}**")

    st.subheader("📌 Kết luận nghiệp vụ")

    if score >= 75:
        st.error("Khuyến nghị kiểm tra thực tế container hoặc soi chiếu lại với góc khác.")
    elif score >= 45:
        st.warning("Khuyến nghị soi chiếu tăng cường và rà soát hồ sơ.")
    else:
        st.success("Chưa phát hiện dấu hiệu bất thường rõ ràng.")
else:
    st.info("Vui lòng nhập Manifest và upload ảnh X-ray.")