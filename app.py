import streamlit as st
import numpy as np
from PIL import Image
import cv2
import re

st.set_page_config(
    page_title="X-Ray Cargo Analyzer V7.1",
    layout="wide"
)

st.title("📦 X-Ray Cargo Analyzer V7.1 LOCAL PROFESSIONAL")
st.caption("Phân tích X-ray container bằng OpenCV local, không dùng API, không tốn phí.")

manifest = st.text_area(
    "📄 Manifest / Khai báo hàng hóa",
    height=130,
    placeholder="Ví dụ: TOTAL:864 CARTONS CERAMIC FLOWERPOT HS CODE:691390"
)

uploaded = st.file_uploader(
    "📤 Upload ảnh X-ray",
    type=["jpg", "jpeg", "png"]
)

GROUP_KEYWORDS = {
    "ceramic": [
        "ceramic", "flowerpot", "flower pot", "pottery", "porcelain",
        "tile", "tiles", "brick", "bricks", "gạch", "gach", "gốm", "gom",
        "stone", "granite", "marble"
    ],
    "machinery": [
        "machine", "machinery", "equipment", "engine", "motor", "forklift",
        "compressor", "generator", "pump", "bearing", "steel", "iron",
        "metal", "pipe", "tool", "parts", "crane", "excavator"
    ],
    "light": [
        "plastic", "toy", "toys", "paper", "carton", "cartons", "foam",
        "polystyrene", "bag", "bags"
    ],
    "textile": [
        "textile", "garment", "clothes", "fabric", "cotton", "shoe", "shoes"
    ],
    "electronics": [
        "battery", "lithium", "electronics", "phone", "computer", "adapter",
        "charger", "circuit", "pcb"
    ],
    "food": [
        "food", "fruit", "vegetable", "seafood", "meat", "organic"
    ],
    "chemical": [
        "chemical", "powder", "liquid", "paint", "solvent", "resin"
    ]
}

EXPECTED_PROFILE = {
    "ceramic": {"density": "medium", "uniform": True, "mechanical": False},
    "machinery": {"density": "dense", "uniform": False, "mechanical": True},
    "light": {"density": "light", "uniform": False, "mechanical": False},
    "textile": {"density": "light", "uniform": False, "mechanical": False},
    "electronics": {"density": "dense", "uniform": False, "mechanical": True},
    "food": {"density": "medium", "uniform": False, "mechanical": False},
    "chemical": {"density": "medium", "uniform": True, "mechanical": False},
    "unknown": {"density": "unknown", "uniform": False, "mechanical": False}
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9À-ỹ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_manifest(text):
    text = clean_text(text)
    scores = {g: 0 for g in GROUP_KEYWORDS}

    for group, words in GROUP_KEYWORDS.items():
        for w in words:
            if w in text:
                scores[group] += 1

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "unknown", scores

    return best, scores


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def normalize_gray(gray):
    p2, p98 = np.percentile(gray, (2, 98))
    clipped = np.clip(gray, p2, p98)
    norm = ((clipped - p2) / (p98 - p2 + 1e-6) * 255).astype(np.uint8)
    return norm


def crop_main_cargo_region(gray):
    h, w = gray.shape
    norm = normalize_gray(gray)

    mask = norm < 225
    kernel = np.ones((5, 5), np.uint8)

    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < h * w * 0.01:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(ch, 1)

        if aspect > 15 or aspect < 0.05:
            continue

        valid.append((area, x, y, cw, ch))

    if not valid:
        return gray, (0, 0, w, h), mask_u8

    valid.sort(reverse=True, key=lambda x: x[0])
    _, x, y, cw, ch = valid[0]

    pad = int(min(h, w) * 0.03)

    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + cw + pad, w)
    y1 = min(y + ch + pad, h)

    crop = gray[y0:y1, x0:x1]

    return crop, (x0, y0, x1, y1), mask_u8


def compute_features(gray_crop):
    norm = normalize_gray(gray_crop)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)

    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    edges = cv2.Canny(blur, 45, 130)
    edge_density = np.sum(edges > 0) / edges.size

    mean_density = float(np.mean(norm))
    std_density = float(np.std(norm))
    variance = float(np.var(norm))

    dark_ratio = float(np.sum(norm < 85) / norm.size)
    very_dark_ratio = float(np.sum(norm < 55) / norm.size)
    bright_ratio = float(np.sum(norm > 210) / norm.size)

    lap = cv2.Laplacian(blur, cv2.CV_64F)
    texture_score = float(np.var(lap))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=35,
        maxLineGap=8
    )

    line_count = 0 if lines is None else len(lines)

    dark_mask = (norm < 95).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)

    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    suspicious = []
    h, w = norm.shape

    for c in contours:
        area = cv2.contourArea(c)

        if area < max(1500, h * w * 0.006):
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(ch, 1)

        if area > h * w * 0.35:
            continue

        if x < 15 or y < 15:
            continue

        if aspect > 6 or aspect < 0.15:
            continue

        region = norm[y:y + ch, x:x + cw]
        local_std = float(np.std(region))

        if local_std < 18:
            continue

        region_dark = float(np.sum(region < 85) / region.size)
        region_std = float(np.std(region))

        if region_dark > 0.18 or region_std > 42:
            suspicious.append({
                "box": (x, y, cw, ch),
                "area": float(area),
                "dark": region_dark,
                "std": region_std,
                "aspect": float(aspect)
            })

    suspicious_count = len(suspicious)

    uniformity_score = 0

    if edge_density < 0.035:
        uniformity_score += 30
    if std_density < 55:
        uniformity_score += 25
    if suspicious_count <= 2:
        uniformity_score += 20
    if texture_score < 120:
        uniformity_score += 25

    mechanical_score = 0

    if edge_density > 0.045:
        mechanical_score += 25
    if line_count > 45:
        mechanical_score += 35
    if suspicious_count >= 3:
        mechanical_score += 30
    if texture_score > 160:
        mechanical_score += 15
    if very_dark_ratio > 0.035:
        mechanical_score += 15

    return {
        "norm": norm,
        "enhanced": enhanced,
        "edges": edges,
        "mean_density": mean_density,
        "std_density": std_density,
        "variance": variance,
        "dark_ratio": dark_ratio,
        "very_dark_ratio": very_dark_ratio,
        "bright_ratio": bright_ratio,
        "edge_density": edge_density,
        "texture_score": texture_score,
        "line_count": line_count,
        "suspicious": suspicious,
        "suspicious_count": suspicious_count,
        "uniformity_score": uniformity_score,
        "mechanical_score": mechanical_score
    }


def predict_image_profile(features):
    scores = {
        "ceramic": 0,
        "machinery": 0,
        "light": 0,
        "textile": 0,
        "electronics": 0,
        "mixed": 0
    }

    edge = features["edge_density"]
    std = features["std_density"]
    dark = features["dark_ratio"]
    very_dark = features["very_dark_ratio"]
    texture = features["texture_score"]
    lines = features["line_count"]
    suspicious = features["suspicious_count"]
    uniformity = features["uniformity_score"]
    mechanical = features["mechanical_score"]

    if uniformity >= 55:
        scores["ceramic"] += 35
    if 0.03 <= dark <= 0.22:
        scores["ceramic"] += 20
    if std < 60:
        scores["ceramic"] += 20
    if edge < 0.055:
        scores["ceramic"] += 15
    if suspicious <= 4:
        scores["ceramic"] += 10

    if mechanical >= 45:
        scores["machinery"] += 45
    if lines > 45:
        scores["machinery"] += 25
    if suspicious >= 3:
        scores["machinery"] += 25
    if texture > 180:
        scores["machinery"] += 10
    if very_dark > 0.04:
        scores["machinery"] += 10

    if dark < 0.05:
        scores["light"] += 35
    if std < 40:
        scores["light"] += 25
    if edge < 0.035:
        scores["light"] += 20

    if 0.025 <= edge <= 0.06 and dark < 0.10 and 35 <= std <= 70:
        scores["textile"] += 40

    if very_dark > 0.05 and suspicious >= 4:
        scores["electronics"] += 35
    if lines > 80 and texture > 180:
        scores["electronics"] += 25

    if mechanical >= 45 and uniformity < 55:
        scores["mixed"] += 35
    if suspicious >= 5 and std > 60:
        scores["mixed"] += 25

    best = max(scores, key=scores.get)

    if scores[best] < 30:
        best = "unknown"

    return best, scores


def calculate_risk(manifest_type, image_type, features):
    score = 0
    reasons = []

    mechanical_score = features["mechanical_score"]
    uniformity_score = features["uniformity_score"]
    suspicious_count = features["suspicious_count"]
    dark = features["dark_ratio"]
    very_dark = features["very_dark_ratio"]
    texture = features["texture_score"]
    edge = features["edge_density"]
    line_count = features["line_count"]
    std = features["std_density"]

    if manifest_type == "unknown":
        score += 25
        reasons.append("Manifest chưa xác định rõ nhóm hàng, cần mô tả cụ thể hơn.")

    if manifest_type != "unknown" and image_type != "unknown":
        if manifest_type != image_type:
            compatible = False

            if manifest_type == "ceramic" and image_type in ["light", "textile"] and uniformity_score > 55:
                compatible = True

            if not compatible:
                score += 35
                reasons.append(f"Ảnh có đặc trưng gần nhóm '{image_type}', chưa khớp hoàn toàn với Manifest '{manifest_type}'.")

    if manifest_type == "ceramic":
        if mechanical_score >= 45:
            score += 35
            reasons.append("Manifest khai ceramic/gạch/gốm nhưng ảnh có dấu hiệu cấu trúc cơ khí hoặc hàng không đồng nhất.")

        if suspicious_count >= 3:
            score += 35
            reasons.append("Có nhiều vùng đậm đáng chú ý trong vùng hàng.")

        if std > 70:
            score += 15
            reasons.append("Mật độ vùng hàng biến thiên cao, không giống hàng ceramic đồng đều.")

        if line_count > 50:
            score += 25
            reasons.append("Manifest khai ceramic nhưng ảnh có nhiều cấu trúc thẳng/cơ khí.")

    if manifest_type in ["light", "textile"]:
        if dark > 0.20 or very_dark > 0.04:
            score += 35
            reasons.append("Manifest khai hàng nhẹ/hàng mềm nhưng ảnh có mật độ hấp thụ cao.")

        if mechanical_score >= 40:
            score += 30
            reasons.append("Ảnh có cấu trúc phức tạp không phù hợp hàng nhẹ/hàng mềm.")

    if manifest_type == "machinery":
        if dark < 0.08 and mechanical_score < 35:
            score += 35
            reasons.append("Manifest khai máy móc/kim loại nhưng ảnh vùng hàng không có mật độ/cấu trúc tương ứng.")

    if manifest_type != "electronics":
        if very_dark > 0.055 and suspicious_count >= 4:
            score += 25
            reasons.append("Có nhiều cụm rất đậm, cần lưu ý khả năng linh kiện/pin/hàng đặc không khai báo.")

    if texture > 220:
        score += 10
        reasons.append("Texture ảnh phức tạp.")

    if line_count > 45:
        score += 25
        reasons.append("Có nhiều đường cấu trúc thẳng, cần đối chiếu khả năng máy móc/khung kim loại.")

    if edge > 0.065:
        score += 10
        reasons.append("Mật độ cạnh cao.")

    if mechanical_score > 55 and suspicious_count >= 2:
        score += 25
        reasons.append("Có cụm cấu trúc cơ khí/mật độ cao đáng chú ý.")

    score = min(score, 100)

    if score >= 75:
        level = "🔴 RỦI RO CAO"
    elif score >= 45:
        level = "🟠 RỦI RO TRUNG BÌNH"
    else:
        level = "🟢 ÍT NGHI VẤN"

    if not reasons:
        reasons.append("Chưa phát hiện dấu hiệu bất thường rõ theo chỉ số hiện tại.")

    return score, level, reasons


def make_heatmap(gray_crop):
    norm = normalize_gray(gray_crop)
    inv = 255 - norm
    heat = cv2.applyColorMap(inv, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return heat


def draw_suspicious(gray_crop, suspicious):
    out = cv2.cvtColor(normalize_gray(gray_crop), cv2.COLOR_GRAY2RGB)

    for s in suspicious:
        x, y, w, h = s["box"]

        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 3)

        cv2.putText(
            out,
            "DANG LUU Y",
            (x, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    return out


if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    st.subheader("1. Ảnh X-ray gốc")
    st.image(img, width="stretch")

    manifest_type, manifest_scores = classify_manifest(manifest)

    gray_full = to_gray(img)
    gray_crop, crop_box, object_mask = crop_main_cargo_region(gray_full)

    features = compute_features(gray_crop)
    image_type, image_scores = predict_image_profile(features)

    risk_score, risk_level, reasons = calculate_risk(
        manifest_type,
        image_type,
        features
    )

    st.subheader("2. Vùng hàng chính đã tách tự động")
    st.image(gray_crop, width="stretch", clamp=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("3. Heatmap mật độ")
        st.image(make_heatmap(gray_crop), width="stretch")

    with col2:
        st.subheader("4. Vùng đậm/cấu trúc đáng lưu ý")
        st.image(draw_suspicious(gray_crop, features["suspicious"]), width="stretch")

    st.subheader("5. Phân tích kỹ thuật nâng cao")

    c1, c2, c3 = st.columns(3)

    c1.metric("Dark Ratio", round(features["dark_ratio"], 3))
    c1.metric("Very Dark", round(features["very_dark_ratio"], 3))
    c1.metric("Std Density", round(features["std_density"], 1))

    c2.metric("Edge Density", round(features["edge_density"], 3))
    c2.metric("Texture", round(features["texture_score"], 1))
    c2.metric("Line Count", features["line_count"])

    c3.metric("Suspicious Zones", features["suspicious_count"])
    c3.metric("Uniformity Score", features["uniformity_score"])
    c3.metric("Mechanical Score", features["mechanical_score"])

    st.subheader("6. Nhận định loại hàng")

    st.write(f"Manifest AI phân loại: **{manifest_type.upper()}**")
    st.write(f"Ảnh AI suy đoán gần nhất: **{image_type.upper()}**")

    with st.expander("Chi tiết điểm Manifest"):
        st.json(manifest_scores)

    with st.expander("Chi tiết điểm ảnh"):
        st.json(image_scores)

    st.subheader("7. Đánh giá rủi ro")

    st.markdown(f"## {risk_level}")
    st.progress(risk_score / 100)
    st.write(f"Điểm nghi vấn: **{risk_score}/100**")

    st.subheader("8. Giải thích")
    for r in reasons:
        st.write("- " + r)

    st.subheader("9. Kết luận nghiệp vụ")

    if risk_score >= 75:
        st.error("Khuyến nghị kiểm tra thực tế hoặc soi chiếu lại bằng góc khác.")
    elif risk_score >= 45:
        st.warning("Khuyến nghị soi chiếu tăng cường và rà soát hồ sơ/chứng từ.")
    else:
        st.success("Chưa phát hiện dấu hiệu bất thường rõ ràng theo chỉ số hiện tại.")

else:
    st.info("Vui lòng nhập Manifest và upload ảnh X-ray để phân tích.")