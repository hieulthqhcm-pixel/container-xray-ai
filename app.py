import streamlit as st
import cv2
import numpy as np
from PIL import Image
import hashlib
import re

st.set_page_config(
    page_title="X-Ray Cargo Analyzer V7.2 LOCAL PROFESSIONAL",
    page_icon="📦",
    layout="wide"
)

st.title("📦 X-Ray Cargo Analyzer V7.2 LOCAL PROFESSIONAL")
st.caption("Phân tích X-ray container bằng OpenCV local, không dùng API, không tốn phí.")

# =========================
# RESET KHI ĐỔI ẢNH / MANIFEST
# =========================

def make_hash(uploaded_file, manifest_text):
    file_hash = ""
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

    text_hash = hashlib.md5(manifest_text.encode("utf-8")).hexdigest()
    return file_hash + "_" + text_hash


def reset_old_result():
    keys = [
        "result",
        "last_key",
        "img_original",
        "img_gray",
        "img_heatmap",
        "img_marked",
        "metrics",
        "manifest_class",
        "image_class",
        "risk_score",
        "risk_level",
        "reasons",
        "conclusion"
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


# =========================
# NHẬN DIỆN MANIFEST
# =========================

def classify_manifest(text):
    t = text.upper()

    machinery_kw = [
        "EXCAVATOR", "MACHINE", "MACHINERY", "FORKLIFT", "TRUCK",
        "ENGINE", "MOTOR", "CRANE", "LOADER", "BULLDOZER",
        "8429", "8427", "8431", "8701", "8704"
    ]

    ceramic_kw = [
        "CERAMIC", "TILE", "PORCELAIN", "STONE", "GRANITE",
        "MARBLE", "BRICK", "6907", "6908"
    ]

    textile_kw = [
        "TEXTILE", "GARMENT", "CLOTH", "FABRIC", "SHIRT",
        "PANTS", "COTTON", "POLYESTER", "APPAREL"
    ]

    plastic_kw = [
        "PLASTIC", "POLY", "PVC", "PE", "PP", "RESIN",
        "POLYSTYRENE", "3901", "3902", "3903"
    ]

    electronics_kw = [
        "ELECTRONIC", "COMPUTER", "LAPTOP", "PHONE",
        "CIRCUIT", "BOARD", "BATTERY", "8507", "8517"
    ]

    scores = {
        "MACHINERY": sum(k in t for k in machinery_kw),
        "CERAMIC": sum(k in t for k in ceramic_kw),
        "TEXTILE": sum(k in t for k in textile_kw),
        "PLASTIC": sum(k in t for k in plastic_kw),
        "ELECTRONICS": sum(k in t for k in electronics_kw),
    }

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "UNKNOWN", scores

    return best, scores


# =========================
# PHÂN TÍCH ẢNH X-RAY
# =========================

def analyze_image(image_pil):
    img = np.array(image_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Tăng tương phản
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Làm mịn nhẹ
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Edge
    edges = cv2.Canny(blur, 50, 150)

    # Threshold vùng đậm
    _, dark_mask = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)
    _, very_dark_mask = cv2.threshold(blur, 55, 255, cv2.THRESH_BINARY_INV)

    total_pixels = gray.size
    dark_ratio = np.sum(dark_mask > 0) / total_pixels
    very_dark_ratio = np.sum(very_dark_mask > 0) / total_pixels
    edge_density = np.sum(edges > 0) / total_pixels
    std_density = float(np.std(gray))
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Đếm đường thẳng
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=40,
        maxLineGap=8
    )
    line_count = 0 if lines is None else len(lines)

    # Tìm vùng nghi vấn
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    suspicious_boxes = []
    h, w = gray.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < total_pixels * 0.003:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw < 25 or bh < 25:
            continue

        box_ratio = area / (bw * bh + 1)
        if box_ratio < 0.15:
            continue

        suspicious_boxes.append((x, y, bw, bh, area))

    suspicious_boxes = sorted(suspicious_boxes, key=lambda b: b[4], reverse=True)[:5]

    # Heatmap
    heat = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    # Ảnh đánh dấu
    marked = img.copy()
    for i, (x, y, bw, bh, area) in enumerate(suspicious_boxes):
        cv2.rectangle(marked, (x, y), (x + bw, y + bh), (255, 0, 0), 3)
        cv2.putText(
            marked,
            f"DANG LUU Y {i+1}",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2
        )

    # Phân loại ảnh theo đặc trưng
    if edge_density > 0.025 and line_count > 120:
        image_class = "MACHINERY"
    elif dark_ratio > 0.20 and edge_density < 0.018:
        image_class = "CERAMIC"
    elif texture < 80 and dark_ratio < 0.12:
        image_class = "TEXTILE"
    elif dark_ratio < 0.18 and texture < 180:
        image_class = "PLASTIC"
    elif edge_density > 0.035 and texture > 300:
        image_class = "ELECTRONICS"
    else:
        image_class = "UNKNOWN"

    metrics = {
        "Dark Ratio": round(float(dark_ratio), 3),
        "Very Dark": round(float(very_dark_ratio), 3),
        "Std Density": round(std_density, 1),
        "Edge Density": round(float(edge_density), 3),
        "Texture": round(texture, 1),
        "Line Count": int(line_count),
        "Suspicious Zones": int(len(suspicious_boxes)),
        "Uniformity Score": round(100 - min(std_density, 100), 1)
    }

    return {
        "original": img,
        "gray": enhanced,
        "heatmap": heat_rgb,
        "marked": marked,
        "metrics": metrics,
        "image_class": image_class,
        "suspicious_boxes": suspicious_boxes
    }


# =========================
# ĐÁNH GIÁ RỦI RO
# =========================

def calculate_risk(manifest_class, image_class, metrics):
    score = 20
    reasons = []

    if manifest_class == "UNKNOWN":
        score += 15
        reasons.append("Manifest chưa đủ rõ để phân loại chắc chắn.")

    if image_class == "UNKNOWN":
        score += 15
        reasons.append("Ảnh X-ray chưa đủ đặc trưng để phân loại chắc chắn.")

    if manifest_class != "UNKNOWN" and image_class != "UNKNOWN":
        if manifest_class != image_class:
            score += 30
            reasons.append(
                f"Ảnh có đặc trưng gần nhóm '{image_class}', chưa khớp hoàn toàn với Manifest '{manifest_class}'."
            )
        else:
            score -= 10
            reasons.append("Đặc trưng ảnh tương đối phù hợp với Manifest khai báo.")

    if metrics["Very Dark"] > 0.06:
        score += 10
        reasons.append("Có vùng rất đậm, cần đối chiếu khả năng che khuất hoặc vật liệu mật độ cao.")

    if metrics["Edge Density"] > 0.025:
        score += 10
        reasons.append("Có nhiều đường/cấu trúc thẳng, phù hợp hàng máy móc hoặc khung kim loại.")

    if metrics["Suspicious Zones"] >= 2:
        score += 10
        reasons.append("Có nhiều vùng đậm/cấu trúc đáng lưu ý.")

    if metrics["Line Count"] > 250:
        score += 8
        reasons.append("Số lượng đường biên lớn, ảnh có cấu trúc phức tạp.")

    score = max(0, min(100, score))

    if score >= 75:
        level = "🔴 RỦI RO CAO"
        conclusion = "Khuyến nghị kiểm tra thực tế hoặc soi chiếu tăng cường, đối chiếu kỹ hồ sơ."
    elif score >= 45:
        level = "🟠 RỦI RO TRUNG BÌNH"
        conclusion = "Khuyến nghị soi chiếu tăng cường và rà soát hồ sơ khai báo."
    else:
        level = "🟢 RỦI RO THẤP"
        conclusion = "Có thể xem xét thông quan nếu hồ sơ đầy đủ và không có dấu hiệu nghiệp vụ khác."

    return score, level, reasons, conclusion


# =========================
# GIAO DIỆN
# =========================

manifest = st.text_area(
    "📄 Manifest / Khai báo hàng hóa",
    height=130,
    placeholder="Ví dụ: USED EXCAVATOR H.S.CODE: 84295200..."
)

uploaded_file = st.file_uploader(
    "📤 Upload ảnh X-ray",
    type=["jpg", "jpeg", "png"]
)

if st.button("🔄 Xóa kết quả cũ / Reset app"):
    reset_old_result()
    st.rerun()

if uploaded_file is not None:
    current_key = make_hash(uploaded_file, manifest)

    if st.session_state.get("last_key") != current_key:
        for k in [
            "result",
            "img_original",
            "img_gray",
            "img_heatmap",
            "img_marked",
            "metrics",
            "manifest_class",
            "image_class",
            "risk_score",
            "risk_level",
            "reasons",
            "conclusion"
        ]:
            if k in st.session_state:
                del st.session_state[k]

        st.session_state["last_key"] = current_key

    image_pil = Image.open(uploaded_file)

    st.subheader("1. Ảnh X-ray gốc")
    st.image(image_pil, use_container_width=True)

    if manifest.strip() == "":
        st.warning("Vui lòng nhập Manifest / khai báo hàng hóa để phân tích đầy đủ.")
        st.stop()

    # Luôn phân tích lại theo ảnh + manifest hiện tại
    manifest_class, manifest_scores = classify_manifest(manifest)
    img_result = analyze_image(image_pil)

    image_class = img_result["image_class"]
    metrics = img_result["metrics"]

    risk_score, risk_level, reasons, conclusion = calculate_risk(
        manifest_class,
        image_class,
        metrics
    )

    st.subheader("2. Vùng hàng chính đã tách tự động")
    st.image(img_result["gray"], use_container_width=True, clamp=True)

    st.subheader("3. Heatmap mật độ")
    st.image(img_result["heatmap"], use_container_width=True)

    st.subheader("4. Vùng đậm/cấu trúc đáng lưu ý")
    st.image(img_result["marked"], use_container_width=True)

    st.subheader("5. Phân tích kỹ thuật nâng cao")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Dark Ratio", metrics["Dark Ratio"])
        st.metric("Very Dark", metrics["Very Dark"])
        st.metric("Std Density", metrics["Std Density"])
        st.metric("Edge Density", metrics["Edge Density"])

    with col2:
        st.metric("Texture", metrics["Texture"])
        st.metric("Line Count", metrics["Line Count"])
        st.metric("Suspicious Zones", metrics["Suspicious Zones"])
        st.metric("Uniformity Score", metrics["Uniformity Score"])

    st.subheader("6. Nhận định loại hàng")

    st.write(f"Manifest AI phân loại: **{manifest_class}**")
    st.write(f"Ảnh AI suy đoán gần nhất: **{image_class}**")

    with st.expander("Chi tiết điểm Manifest"):
        st.json(manifest_scores)

    with st.expander("Chi tiết điểm ảnh"):
        st.json(metrics)

    st.subheader("7. Đánh giá rủi ro")
    st.markdown(f"## {risk_level}")
    st.progress(risk_score / 100)
    st.write(f"Điểm nghi vấn: **{risk_score}/100**")

    st.subheader("8. Giải thích")
    if reasons:
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.write("- Chưa phát hiện dấu hiệu nổi bật.")

    st.subheader("9. Kết luận nghiệp vụ")
    st.warning(conclusion)

else:
    st.info("Hãy upload ảnh X-ray để bắt đầu phân tích.")