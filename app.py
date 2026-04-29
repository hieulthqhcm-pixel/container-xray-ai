# =========================
# THAY TOÀN BỘ app.py
# X-RAY CARGO ANALYZER V7.5 AI PROFESSIONAL
# =========================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import hashlib

# =========================
# CẤU HÌNH APP
# =========================

st.set_page_config(
    page_title="X-Ray Cargo Analyzer V7.5 AI PROFESSIONAL",
    page_icon="📦",
    layout="wide"
)

st.title("📦 X-Ray Cargo Analyzer V7.5 AI PROFESSIONAL")

st.caption(
    "AI phân tích ảnh soi chiếu container bằng OpenCV local - không dùng API."
)

# =========================
# RESET SESSION
# =========================

def make_hash(uploaded_file, manifest_text):

    file_hash = ""

    if uploaded_file is not None:

        file_hash = hashlib.md5(
            uploaded_file.getvalue()
        ).hexdigest()

    text_hash = hashlib.md5(
        manifest_text.encode("utf-8")
    ).hexdigest()

    return file_hash + "_" + text_hash


def clear_old_results():

    keys = [
        "last_key"
    ]

    for k in keys:

        if k in st.session_state:
            del st.session_state[k]


# =========================
# PHÂN LOẠI MANIFEST
# =========================

def classify_manifest(text):

    t = text.upper()

    groups = {

        "MACHINERY": [
            "EXCAVATOR",
            "USED EXCAVATOR",
            "MACHINE",
            "MACHINERY",
            "FORKLIFT",
            "TRUCK",
            "ENGINE",
            "MOTOR",
            "CRANE",
            "LOADER",
            "BULLDOZER",
            "8429",
            "842952",
            "84295200",
            "8427",
            "8431",
            "8701",
            "8704"
        ],

        "CERAMIC": [
            "CERAMIC",
            "TILE",
            "PORCELAIN",
            "STONE",
            "GRANITE",
            "MARBLE",
            "BRICK",
            "6907",
            "6908"
        ],

        "TEXTILE": [
            "TEXTILE",
            "GARMENT",
            "CLOTH",
            "FABRIC",
            "SHIRT",
            "PANTS",
            "COTTON",
            "POLYESTER",
            "APPAREL"
        ],

        "PLASTIC": [
            "PLASTIC",
            "POLY",
            "PVC",
            "PE",
            "PP",
            "RESIN",
            "POLYSTYRENE",
            "3901",
            "3902",
            "3903",
            "390311"
        ],

        "ELECTRONICS": [
            "ELECTRONIC",
            "COMPUTER",
            "LAPTOP",
            "PHONE",
            "CIRCUIT",
            "BOARD",
            "BATTERY",
            "8507",
            "8517"
        ]
    }

    scores = {}

    for group, keywords in groups.items():

        scores[group] = sum(
            1 for kw in keywords if kw in t
        )

    best = max(scores, key=scores.get)

    if scores[best] == 0:

        return "UNKNOWN", scores

    return best, scores


# =========================
# AI PHÂN TÍCH ẢNH X-RAY
# =========================

def analyze_image(image_pil):

    img = np.array(
        image_pil.convert("RGB")
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    h, w = gray.shape

    total_pixels = gray.size

    # =========================
    # ROI
    # =========================

    roi_x1 = int(w * 0.18)
    roi_x2 = int(w * 0.82)

    roi_y1 = int(h * 0.18)
    roi_y2 = int(h * 0.82)

    roi_mask = np.zeros_like(gray)

    roi_mask[
        roi_y1:roi_y2,
        roi_x1:roi_x2
    ] = 255

    # =========================
    # CLAHE
    # =========================

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0
    )

    # =========================
    # EDGE
    # =========================

    edges = cv2.Canny(
        blur,
        35,
        120
    )

    edges = cv2.bitwise_and(
        edges,
        roi_mask
    )

    # =========================
    # DARK MASK
    # =========================

    dark_threshold = np.percentile(
        blur,
        28
    )

    dark_mask = np.where(
        blur < dark_threshold,
        255,
        0
    ).astype(np.uint8)

    dark_mask = cv2.bitwise_and(
        dark_mask,
        roi_mask
    )

    # =========================
    # OBJECT MASK
    # =========================

    edge_dilate = cv2.dilate(
        edges,
        np.ones((5, 5), np.uint8),
        iterations=1
    )

    object_mask = cv2.bitwise_and(
        dark_mask,
        edge_dilate
    )

    kernel = np.ones((5, 5), np.uint8)

    object_mask = cv2.morphologyEx(
        object_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    object_mask = cv2.morphologyEx(
        object_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # =========================
    # CONTOUR
    # =========================

    contours, _ = cv2.findContours(
        object_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    suspicious_boxes = []

    machinery_shapes = 0
    arm_shapes = 0
    cabin_shapes = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < total_pixels * 0.002:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # bỏ box nhỏ
        if bw < 45 or bh < 45:
            continue

        # bỏ chữ dọc
        if bh > bw * 3:
            continue

        # bỏ sát mép
        if x < roi_x1 or x + bw > roi_x2:
            continue

        if y < roi_y1 or y + bh > roi_y2:
            continue

        roi_edge = edges[
            y:y+bh,
            x:x+bw
        ]

        edge_density_inside = np.sum(
            roi_edge > 0
        ) / (bw * bh)

        if edge_density_inside < 0.02:
            continue

        # =========================
        # SHAPE ANALYSIS
        # =========================

        aspect_ratio = bw / (bh + 1)

        # cabin/body
        if 0.7 < aspect_ratio < 2.5:
            cabin_shapes += 1

        # arm/boom
        if aspect_ratio > 2.8:
            arm_shapes += 1

        machinery_shapes += 1

        suspicious_boxes.append(
            (
                x,
                y,
                bw,
                bh,
                area,
                edge_density_inside
            )
        )

    suspicious_boxes = sorted(
        suspicious_boxes,
        key=lambda b: b[4] * b[5],
        reverse=True
    )[:5]

    # =========================
    # HEATMAP
    # =========================

    density_map = 255 - enhanced

    density_map = cv2.GaussianBlur(
        density_map,
        (9, 9),
        0
    )

    heat = cv2.applyColorMap(
        density_map,
        cv2.COLORMAP_JET
    )

    heat_rgb = cv2.cvtColor(
        heat,
        cv2.COLOR_BGR2RGB
    )

    # =========================
    # DRAW BOX
    # =========================

    marked = img.copy()

    for i, (
        x,
        y,
        bw,
        bh,
        area,
        edge_density_inside
    ) in enumerate(suspicious_boxes):

        cv2.rectangle(
            marked,
            (x, y),
            (x + bw, y + bh),
            (255, 0, 0),
            3
        )

        cv2.putText(
            marked,
            f"MACHINERY AREA {i+1}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

    # =========================
    # METRICS
    # =========================

    dark_ratio = np.sum(
        dark_mask > 0
    ) / total_pixels

    very_dark_ratio = np.sum(
        blur < np.percentile(blur, 12)
    ) / total_pixels

    edge_density = np.sum(
        edges > 0
    ) / total_pixels

    std_density = float(
        np.std(gray)
    )

    texture = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        minLineLength=45,
        maxLineGap=10
    )

    line_count = 0 if lines is None else len(lines)

    # =========================
    # AI CLASSIFICATION
    # =========================

    image_class = "UNKNOWN"

    ai_label = "UNKNOWN"

    ai_confidence = 50

    # EXCAVATOR

    if (
        line_count > 100
        and edge_density > 0.020
        and machinery_shapes >= 2
        and arm_shapes >= 1
    ):

        image_class = "MACHINERY"

        ai_label = "USED EXCAVATOR"

        ai_confidence = 90

    # HEAVY MACHINERY

    elif (
        line_count > 70
        and edge_density > 0.018
        and machinery_shapes >= 2
    ):

        image_class = "MACHINERY"

        ai_label = "HEAVY MACHINERY"

        ai_confidence = 80

    # CERAMIC

    elif (
        dark_ratio > 0.22
        and edge_density < 0.018
    ):

        image_class = "CERAMIC"

        ai_label = "CERAMIC / TILE"

        ai_confidence = 75

    # TEXTILE

    elif (
        texture < 80
        and dark_ratio < 0.14
    ):

        image_class = "TEXTILE"

        ai_label = "TEXTILE"

        ai_confidence = 70

    # PLASTIC

    elif (
        dark_ratio < 0.18
        and texture < 180
    ):

        image_class = "PLASTIC"

        ai_label = "PLASTIC GOODS"

        ai_confidence = 70

    # ELECTRONICS

    elif (
        edge_density > 0.035
        and texture > 300
    ):

        image_class = "ELECTRONICS"

        ai_label = "ELECTRONIC EQUIPMENT"

        ai_confidence = 75

    metrics = {

        "Dark Ratio":
            round(float(dark_ratio), 3),

        "Very Dark":
            round(float(very_dark_ratio), 3),

        "Std Density":
            round(std_density, 1),

        "Edge Density":
            round(float(edge_density), 3),

        "Texture":
            round(texture, 1),

        "Line Count":
            int(line_count),

        "Suspicious Zones":
            int(len(suspicious_boxes)),

        "Machinery Shapes":
            int(machinery_shapes),

        "Arm Shapes":
            int(arm_shapes),

        "Cabin Shapes":
            int(cabin_shapes),

        "AI Confidence":
            int(ai_confidence),

        "Uniformity Score":
            round(
                100 - min(std_density, 100),
                1
            )
    }

    return {

        "original": img,

        "gray": enhanced,

        "heatmap": heat_rgb,

        "marked": marked,

        "metrics": metrics,

        "image_class": image_class,

        "ai_label": ai_label,

        "ai_confidence": ai_confidence,

        "suspicious_boxes": suspicious_boxes
    }


# =========================
# ĐÁNH GIÁ RỦI RO
# =========================

def calculate_risk(
    manifest_class,
    image_class,
    metrics
):

    score = 20

    reasons = []

    if manifest_class == "UNKNOWN":

        score += 15

        reasons.append(
            "Manifest chưa đủ rõ để phân loại chắc chắn."
        )

    if image_class == "UNKNOWN":

        score += 15

        reasons.append(
            "Ảnh X-ray chưa đủ đặc trưng để phân loại chắc chắn."
        )

    if (
        manifest_class != "UNKNOWN"
        and
        image_class != "UNKNOWN"
    ):

        if manifest_class != image_class:

            score += 30

            reasons.append(
                f"Ảnh có đặc trưng gần nhóm '{image_class}', chưa khớp hoàn toàn với Manifest '{manifest_class}'."
            )

        else:

            score -= 10

            reasons.append(
                "Đặc trưng ảnh phù hợp với Manifest khai báo."
            )

    if metrics["Very Dark"] > 0.06:

        score += 10

        reasons.append(
            "Có vùng đậm/mật độ cao cần đối chiếu."
        )

    if metrics["Edge Density"] > 0.025:

        score += 10

        reasons.append(
            "Có nhiều cấu trúc kim loại/cơ khí."
        )

    if metrics["Suspicious Zones"] >= 2:

        score += 8

        reasons.append(
            "Có nhiều vùng cấu trúc đáng lưu ý."
        )

    if metrics["Line Count"] > 250:

        score += 8

        reasons.append(
            "Ảnh có cấu trúc phức tạp."
        )

    score = max(
        0,
        min(100, score)
    )

    if score >= 75:

        level = "🔴 RỦI RO CAO"

        conclusion = (
            "Khuyến nghị kiểm tra thực tế hoặc soi chiếu tăng cường."
        )

    elif score >= 45:

        level = "🟠 RỦI RO TRUNG BÌNH"

        conclusion = (
            "Khuyến nghị soi chiếu tăng cường và rà soát hồ sơ."
        )

    else:

        level = "🟢 RỦI RO THẤP"

        conclusion = (
            "Có thể xem xét thông quan nếu hồ sơ đầy đủ."
        )

    return (
        score,
        level,
        reasons,
        conclusion
    )


# =========================
# GIAO DIỆN
# =========================

manifest = st.text_area(
    "📄 Manifest / Khai báo hàng hóa",
    height=130,
    placeholder="Ví dụ: 4 UNITS USED EXCAVATOR H.S.CODE: 84295200..."
)

uploaded_file = st.file_uploader(
    "📤 Upload ảnh X-ray",
    type=["jpg", "jpeg", "png"]
)

if st.button("🔄 Reset / Xóa kết quả cũ"):

    clear_old_results()

    st.rerun()

if uploaded_file is None:

    st.info(
        "Hãy upload ảnh X-ray để bắt đầu phân tích."
    )

    st.stop()

current_key = make_hash(
    uploaded_file,
    manifest
)

if st.session_state.get("last_key") != current_key:

    clear_old_results()

    st.session_state["last_key"] = current_key

image_pil = Image.open(uploaded_file)

st.subheader("1. Ảnh X-ray gốc")

st.image(
    image_pil,
    use_container_width=True
)

if manifest.strip() == "":

    st.warning(
        "Vui lòng nhập Manifest / khai báo hàng hóa."
    )

    st.stop()

# =========================
# AI PHÂN TÍCH
# =========================

manifest_class, manifest_scores = classify_manifest(
    manifest
)

img_result = analyze_image(
    image_pil
)

image_class = img_result["image_class"]

metrics = img_result["metrics"]

ai_label = img_result["ai_label"]

ai_confidence = img_result["ai_confidence"]

(
    risk_score,
    risk_level,
    reasons,
    conclusion
) = calculate_risk(
    manifest_class,
    image_class,
    metrics
)

# =========================
# HIỂN THỊ
# =========================

st.subheader(
    "2. Ảnh tăng tương phản"
)

st.image(
    img_result["gray"],
    use_container_width=True,
    clamp=True
)

st.subheader(
    "3. Heatmap mật độ"
)

st.caption(
    "Màu nóng thể hiện vùng đậm/mật độ cao."
)

st.image(
    img_result["heatmap"],
    use_container_width=True
)

st.subheader(
    "4. Vùng cấu trúc cần đối chiếu"
)

st.image(
    img_result["marked"],
    use_container_width=True
)

st.subheader(
    "5. Phân tích kỹ thuật nâng cao"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Dark Ratio",
        metrics["Dark Ratio"]
    )

    st.metric(
        "Very Dark",
        metrics["Very Dark"]
    )

    st.metric(
        "Std Density",
        metrics["Std Density"]
    )

    st.metric(
        "Edge Density",
        metrics["Edge Density"]
    )

with col2:

    st.metric(
        "Texture",
        metrics["Texture"]
    )

    st.metric(
        "Line Count",
        metrics["Line Count"]
    )

    st.metric(
        "Suspicious Zones",
        metrics["Suspicious Zones"]
    )

    st.metric(
        "AI Confidence",
        metrics["AI Confidence"]
    )

st.subheader(
    "6. Nhận định loại hàng"
)

st.write(
    f"Manifest AI phân loại: **{manifest_class}**"
)

st.write(
    f"Ảnh AI suy đoán nhóm: **{image_class}**"
)

st.success(
    f"AI nhận dạng gần đúng: {ai_label}"
)

st.progress(
    ai_confidence / 100
)

st.write(
    f"Độ tin cậy AI: **{ai_confidence}%**"
)

# =========================
# GIẢI THÍCH AI
# =========================

if ai_label == "USED EXCAVATOR":

    st.info(
        """
        AI phát hiện:

        - Cấu trúc tay cần dài
        - Cabin máy công trình
        - Cụm cơ khí lặp lại
        - Mật độ kim loại cao
        - Đặc trưng giống máy xúc đã qua sử dụng
        """
    )

elif ai_label == "HEAVY MACHINERY":

    st.info(
        """
        AI phát hiện:

        - Nhiều cấu trúc cơ khí
        - Nhiều cạnh kim loại
        - Dạng hàng máy móc công trình
        """
    )

elif ai_label == "CERAMIC / TILE":

    st.info(
        """
        AI phát hiện:

        - Mật độ đậm đồng đều
        - Ít cấu trúc cơ khí
        """
    )

elif ai_label == "TEXTILE":

    st.info(
        """
        AI phát hiện:

        - Kết cấu mềm
        - Ít cạnh kim loại
        """
    )

elif ai_label == "PLASTIC GOODS":

    st.info(
        """
        AI phát hiện:

        - Mật độ trung bình
        - Ít vùng kim loại đậm
        """
    )

elif ai_label == "ELECTRONIC EQUIPMENT":

    st.info(
        """
        AI phát hiện:

        - Texture phức tạp
        - Nhiều cấu trúc nhỏ
        """
    )

with st.expander(
    "Chi tiết điểm Manifest"
):

    st.json(
        manifest_scores
    )

with st.expander(
    "Chi tiết phân tích ảnh"
):

    st.json(
        metrics
    )

st.subheader(
    "7. Đánh giá rủi ro"
)

st.markdown(
    f"## {risk_level}"
)

st.progress(
    risk_score / 100
)

st.write(
    f"Điểm nghi vấn: **{risk_score}/100**"
)

st.subheader(
    "8. Giải thích"
)

if reasons:

    for r in reasons:

        st.write(
            f"- {r}"
        )

else:

    st.write(
        "- Chưa phát hiện dấu hiệu nổi bật."
    )

st.subheader(
    "9. Kết luận nghiệp vụ"
)

st.warning(
    conclusion
)