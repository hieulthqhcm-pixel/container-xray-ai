import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="Container X-ray Risk Intelligence V3",
    layout="wide"
)

st.title("Container X-ray Risk Intelligence V3")
st.caption("Phân tích ảnh X-ray container, heatmap mật độ, suy đoán nhóm hàng và đối chiếu Manifest.")

manifest = st.text_area(
    "Nhập Manifest / khai báo hàng hóa",
    placeholder="Ví dụ: ceramic tiles, bricks, machinery, plastic toys, textile, electronics..."
)

uploaded = st.file_uploader(
    "Tải ảnh X-ray container",
    type=["jpg", "jpeg", "png", "bmp"]
)


def classify_manifest(text):
    text = text.lower()

    groups = {
        "brick": ["brick", "bricks", "gạch", "gach", "ceramic", "tile", "tiles", "stone", "marble", "granite", "cement"],
        "machinery": ["machine", "machinery", "equipment", "engine", "motor", "forklift", "steel", "iron", "metal", "tool", "parts"],
        "textile": ["clothes", "garment", "textile", "fabric", "shoes", "cotton", "bag", "bags"],
        "light": ["plastic", "toy", "toys", "paper", "carton", "foam", "polystyrene"],
        "electronics": ["battery", "lithium", "electronics", "computer", "phone", "adapter", "charger", "circuit"],
        "wood": ["wood", "furniture", "chair", "table", "cabinet", "sofa"],
        "food": ["food", "fruit", "vegetable", "seafood", "meat", "organic"],
        "chemical": ["chemical", "powder", "liquid", "paint", "solvent", "resin"]
    }

    matched = []
    for group, words in groups.items():
        if any(w in text for w in words):
            matched.append(group)

    return matched if matched else ["unknown"]


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    crop_margin_x = int(w * 0.04)
    crop_margin_y = int(h * 0.04)

    roi = gray[
        crop_margin_y:h - crop_margin_y,
        crop_margin_x:w - crop_margin_x
    ]

    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi)

    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return gray, roi, enhanced, blur


def create_heatmap(roi):
    norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)

    # Đảo màu: vùng đậm X-ray thành vùng nóng
    inverted = 255 - norm

    heat = cv2.applyColorMap(inverted.astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(roi_rgb, 0.55, heat_rgb, 0.45, 0)

    return heat_rgb, overlay


def analyze_xray(img):
    gray, roi, enhanced, blur = preprocess_image(img)

    edges = cv2.Canny(blur, 40, 120)

    edge_ratio = float(np.count_nonzero(edges) / edges.size)
    mean_density = float(np.mean(roi))
    std_density = float(np.std(roi))

    dark_mask = roi < 75
    very_dark_mask = roi < 45
    bright_mask = roi > 210

    dark_ratio = float(np.count_nonzero(dark_mask) / dark_mask.size)
    very_dark_ratio = float(np.count_nonzero(very_dark_mask) / very_dark_mask.size)
    bright_ratio = float(np.count_nonzero(bright_mask) / bright_mask.size)

    lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())

    kernel = np.ones((5, 5), np.uint8)
    dark_clean = cv2.morphologyEx(
        dark_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        dark_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    suspicious_regions = 0
    large_dark_regions = 0
    region_areas = []

    h, w = roi.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 1200:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Loại nhiễu mép ảnh
        if x < 10 or y < 10 or x + bw > w - 10 or y + bh > h - 10:
            continue

        ratio = bw / max(bh, 1)

        # Loại chữ/mép dài mỏng
        if ratio > 8 or ratio < 0.12:
            continue

        suspicious_regions += 1
        region_areas.append(area)

        if area > 5000:
            large_dark_regions += 1

    region_area_ratio = float(sum(region_areas) / roi.size) if region_areas else 0.0

    heatmap, overlay = create_heatmap(roi)

    return {
        "gray": gray,
        "roi": roi,
        "enhanced": enhanced,
        "edges": edges,
        "heatmap": heatmap,
        "overlay": overlay,
        "edge_ratio": edge_ratio,
        "mean_density": mean_density,
        "std_density": std_density,
        "dark_ratio": dark_ratio,
        "very_dark_ratio": very_dark_ratio,
        "bright_ratio": bright_ratio,
        "lap_var": lap_var,
        "suspicious_regions": suspicious_regions,
        "large_dark_regions": large_dark_regions,
        "region_area_ratio": region_area_ratio
    }


def predict_cargo_type(a):
    edge = a["edge_ratio"]
    std = a["std_density"]
    dark = a["dark_ratio"]
    very_dark = a["very_dark_ratio"]
    bright = a["bright_ratio"]
    regions = a["suspicious_regions"]
    large_regions = a["large_dark_regions"]
    lap = a["lap_var"]

    scores = {
        "brick": 0,
        "machinery": 0,
        "textile": 0,
        "light": 0,
        "electronics": 0,
        "wood": 0,
        "food": 0,
        "mixed": 0,
        "unknown": 0
    }

    # Gạch/vật liệu xây dựng: thường tương đối đồng đều, ít cạnh phức tạp
    if edge < 0.050:
        scores["brick"] += 28
    if std < 42:
        scores["brick"] += 25
    if dark < 0.14:
        scores["brick"] += 18
    if regions <= 2:
        scores["brick"] += 12

    # Máy móc/kim loại: nhiều cạnh, nhiều vùng đậm, density biến thiên
    if edge > 0.050:
        scores["machinery"] += 25
    if std > 36:
        scores["machinery"] += 25
    if dark > 0.09:
        scores["machinery"] += 25
    if regions >= 2:
        scores["machinery"] += 18
    if large_regions >= 1:
        scores["machinery"] += 15
    if lap > 260:
        scores["machinery"] += 12

    # Hàng nhẹ/nhựa/carton
    if dark < 0.07:
        scores["light"] += 30
    if std < 32:
        scores["light"] += 22
    if bright > 0.25:
        scores["light"] += 15

    # Dệt may
    if dark < 0.10 and 30 <= std <= 55:
        scores["textile"] += 25
    if 0.030 <= edge <= 0.075:
        scores["textile"] += 20

    # Điện tử/pin/linh kiện
    if very_dark > 0.025:
        scores["electronics"] += 25
    if regions >= 4:
        scores["electronics"] += 25
    if edge > 0.060:
        scores["electronics"] += 15
    if dark > 0.16:
        scores["electronics"] += 12

    # Gỗ/nội thất
    if 0.045 <= edge <= 0.090 and 35 <= std <= 60 and dark < 0.16:
        scores["wood"] += 30

    # Thực phẩm/hữu cơ
    if dark < 0.11 and 35 <= std <= 65 and bright < 0.35:
        scores["food"] += 18

    # Hàng hỗn hợp
    if edge > 0.070 and std > 43:
        scores["mixed"] += 35
    if dark > 0.15 and bright > 0.12:
        scores["mixed"] += 22
    if regions >= 3:
        scores["mixed"] += 18
    if lap > 350:
        scores["mixed"] += 15

    predicted = max(scores, key=scores.get)
    confidence = scores[predicted]

    if confidence < 35:
        predicted = "unknown"

    return predicted, scores


def calculate_risk(manifest_groups, predicted_type, cargo_scores, a):
    risks = []
    recommendations = []
    risk_score = 0

    edge = a["edge_ratio"]
    std = a["std_density"]
    dark = a["dark_ratio"]
    very_dark = a["very_dark_ratio"]
    bright = a["bright_ratio"]
    regions = a["suspicious_regions"]
    large_regions = a["large_dark_regions"]
    region_area_ratio = a["region_area_ratio"]

    manifest_unknown = "unknown" in manifest_groups

    if manifest_unknown:
        risks.append("Manifest chưa xác định rõ nhóm hàng. Cần nhập mô tả cụ thể hơn.")
        risk_score += 20
        recommendations.append("Bổ sung tên hàng chi tiết, số lượng, chất liệu, công dụng, HS code nếu có.")

    if predicted_type != "unknown" and not manifest_unknown and predicted_type not in manifest_groups:
        risks.append(f"AI suy đoán ảnh giống nhóm '{predicted_type}' nhưng Manifest khai {manifest_groups}.")
        risk_score += 45

    low_density_declared = any(g in manifest_groups for g in ["brick", "light", "textile", "food"])
    heavy_image_sign = predicted_type in ["machinery", "electronics", "mixed"] or dark > 0.11 or edge > 0.055

    if low_density_declared and heavy_image_sign:
        risks.append("Manifest khai nhóm hàng có tính đồng đều/nhẹ hơn, nhưng ảnh có dấu hiệu hàng đặc, kim loại hoặc cấu trúc phức tạp.")
        risk_score += 35
        recommendations.append("Kiểm tra trực quan ảnh X-ray, đối chiếu trọng lượng, kích thước kiện và chứng từ.")

    if "brick" in manifest_groups:
        if edge > 0.050:
            risks.append("Khai gạch/vật liệu xây dựng nhưng ảnh có nhiều cấu trúc cạnh hơn mức kỳ vọng của hàng xếp đều.")
            risk_score += 18
        if std > 42:
            risks.append("Khai gạch nhưng mật độ ảnh biến thiên cao, không giống hàng đồng nhất.")
            risk_score += 20
        if dark > 0.12:
            risks.append("Khai gạch nhưng có vùng hấp thụ tia X mạnh, cần kiểm tra khả năng lẫn hàng kim loại/máy móc.")
            risk_score += 22

    if "machinery" not in manifest_groups and predicted_type == "machinery":
        risks.append("Manifest không khai máy móc nhưng ảnh có đặc trưng máy móc/kim loại.")
        risk_score += 35
        recommendations.append("Đề xuất kiểm tra thực tế hoặc soi chiếu bổ sung ở góc khác.")

    if "electronics" not in manifest_groups and (very_dark > 0.035 and regions >= 3):
        risks.append("Có nhiều cụm đậm nhỏ, cần lưu ý khả năng linh kiện, pin hoặc hàng đặc không khai báo.")
        risk_score += 25

    if large_regions >= 2:
        risks.append("Có nhiều vùng hấp thụ tia X lớn.")
        risk_score += 18

    if region_area_ratio > 0.18:
        risks.append("Tổng diện tích vùng đậm lớn so với ảnh, cần kiểm tra hàng đặc hoặc hàng che giấu.")
        risk_score += 18

    if bright > 0.45:
        risks.append("Ảnh có nhiều vùng sáng/rỗng, cần kiểm tra bố trí hàng hoặc khả năng khai thiếu hàng.")
        risk_score += 12

    if not risks:
        risks.append("Chưa phát hiện mâu thuẫn rõ theo các chỉ số hiện tại.")
        recommendations.append("Vẫn cần cán bộ soi chiếu đánh giá trực quan trước khi kết luận.")
        risk_score = 10

    risk_score = min(int(risk_score), 100)

    if risk_score >= 75:
        level = "RỦI RO CAO"
    elif risk_score >= 45:
        level = "RỦI RO TRUNG BÌNH"
    else:
        level = "RỦI RO THẤP"

    if risk_score >= 75:
        recommendations.append("Ưu tiên kiểm tra thực tế hoặc soi chiếu lại với góc khác.")
    elif risk_score >= 45:
        recommendations.append("Cần cán bộ soi chiếu rà soát kỹ các vùng đậm và đối chiếu chứng từ.")
    else:
        recommendations.append("Có thể xử lý theo luồng bình thường nếu các thông tin khác phù hợp.")

    return risk_score, level, risks, recommendations


def level_badge(level):
    if level == "RỦI RO CAO":
        st.error(level)
    elif level == "RỦI RO TRUNG BÌNH":
        st.warning(level)
    else:
        st.success(level)


if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    manifest_groups = classify_manifest(manifest)
    analysis = analyze_xray(img)
    predicted_type, cargo_scores = predict_cargo_type(analysis)
    risk_score, level, risks, recommendations = calculate_risk(
        manifest_groups,
        predicted_type,
        cargo_scores,
        analysis
    )

    st.subheader("1. Ảnh gốc")
    st.image(img, width="stretch")

    st.subheader("2. Heatmap mật độ X-ray")
    st.caption("Heatmap chỉ thể hiện vùng hấp thụ tia X mạnh/yếu, không phải kết luận hàng cấm.")
    st.image(analysis["overlay"], width="stretch")

    st.subheader("3. Chỉ số kỹ thuật ảnh")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Chỉ số cạnh/cấu trúc", f"{analysis['edge_ratio']:.3f}")
        st.metric("Độ biến thiên mật độ", f"{analysis['std_density']:.1f}")
        st.metric("Tỷ lệ vùng đậm", f"{analysis['dark_ratio']:.2%}")
        st.metric("Tỷ lệ vùng rất đậm", f"{analysis['very_dark_ratio']:.2%}")

    with col2:
        st.metric("Mật độ trung bình", f"{analysis['mean_density']:.1f}")
        st.metric("Vùng sáng/rỗng", f"{analysis['bright_ratio']:.2%}")
        st.metric("Số vùng đậm đáng chú ý", analysis["suspicious_regions"])
        st.metric("Vùng đậm lớn", analysis["large_dark_regions"])

    st.subheader("4. AI suy đoán nhóm hàng từ ảnh")
    st.write(f"Nhóm ảnh giống nhất: **{predicted_type}**")
    st.json(cargo_scores)

    st.subheader("5. Nhóm hàng theo Manifest")
    st.write(manifest_groups)

    st.subheader("6. Đánh giá rủi ro")
    st.metric("Risk Score", f"{risk_score}/100")
    level_badge(level)

    st.subheader("7. Lý do đánh giá")
    for r in risks:
        st.write(f"- {r}")

    st.subheader("8. Khuyến nghị xử lý")
    for rec in recommendations:
        st.write(f"- {rec}")

    st.subheader("9. Ảnh tăng tương phản")
    st.image(analysis["enhanced"], width="stretch", clamp=True)

    st.subheader("10. Ảnh cạnh/cấu trúc")
    st.image(analysis["edges"], width="stretch", clamp=True)

    st.caption(f"Báo cáo tạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

else:
    st.info("Nhập Manifest và tải ảnh X-ray container để bắt đầu phân tích.")