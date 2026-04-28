import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI X-ray Container Risk Analyzer",
    layout="wide"
)

st.title("AI phân tích X-ray container và đối chiếu Manifest")
st.caption("Phiên bản nâng cấp: phân tích mật độ, cấu trúc, độ đồng nhất và suy đoán nhóm hàng.")

manifest = st.text_area(
    "Nhập Manifest / khai báo hàng hóa",
    placeholder="Ví dụ: ceramic tiles, bricks, machinery, plastic toys, textile, furniture..."
)

uploaded = st.file_uploader(
    "Tải ảnh X-ray container",
    type=["jpg", "jpeg", "png", "bmp"]
)


def classify_manifest(text):
    text = text.lower()

    groups = {
        "brick": ["brick", "bricks", "gạch", "ceramic", "tile", "tiles", "stone", "marble", "granite"],
        "machinery": ["machine", "machinery", "equipment", "engine", "motor", "forklift", "steel", "iron", "metal", "tool"],
        "textile": ["clothes", "garment", "textile", "fabric", "shoes", "cotton", "bag"],
        "light": ["plastic", "toy", "toys", "paper", "carton", "foam", "polystyrene"],
        "electronics": ["battery", "lithium", "electronics", "computer", "phone", "adapter", "charger"],
        "wood": ["wood", "furniture", "chair", "table", "cabinet", "sofa"],
        "food": ["food", "fruit", "vegetable", "seafood", "meat", "organic"]
    }

    matched = []

    for group, words in groups.items():
        if any(w in text for w in words):
            matched.append(group)

    return matched if matched else ["unknown"]


def analyze_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 120)
    edge_ratio = np.count_nonzero(edges) / edges.size

    mean_density = float(np.mean(gray))
    std_density = float(np.std(gray))

    dark_mask = gray < 75
    very_dark_mask = gray < 45
    bright_mask = gray > 210

    dark_ratio = np.count_nonzero(dark_mask) / dark_mask.size
    very_dark_ratio = np.count_nonzero(very_dark_mask) / very_dark_mask.size
    bright_ratio = np.count_nonzero(bright_mask) / bright_mask.size

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    kernel = np.ones((5, 5), np.uint8)
    dark_clean = cv2.morphologyEx(
        dark_mask.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        dark_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    suspicious_regions = 0
    large_dark_regions = 0
    region_areas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 800:
            suspicious_regions += 1
            region_areas.append(area)

            x, y, w, h = cv2.boundingRect(cnt)

            if area > 3000:
                large_dark_regions += 1

            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(
                result,
                "NGHI VAN",
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 0),
                2
            )

    region_area_sum = float(sum(region_areas)) if region_areas else 0.0
    region_area_ratio = region_area_sum / (gray.shape[0] * gray.shape[1])

    return {
        "gray": gray,
        "enhanced": enhanced,
        "edges": edges,
        "result": result,
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
        "mixed": 0
    }

    # Gạch / vật liệu xây dựng: thường đồng đều, dạng khối, ít cấu trúc nhỏ
    if edge < 0.055:
        scores["brick"] += 30
    if std < 42:
        scores["brick"] += 25
    if dark < 0.16:
        scores["brick"] += 15
    if regions <= 2:
        scores["brick"] += 15

    # Máy móc / kim loại: nhiều cạnh, nhiều vùng đậm, mật độ biến thiên
    if edge > 0.055:
        scores["machinery"] += 25
    if std > 38:
        scores["machinery"] += 25
    if dark > 0.10:
        scores["machinery"] += 25
    if regions >= 2:
        scores["machinery"] += 20
    if large_regions >= 1:
        scores["machinery"] += 15

    # Hàng nhẹ: ít vùng đậm, mật độ thấp, tương đối sáng
    if dark < 0.08:
        scores["light"] += 35
    if std < 35:
        scores["light"] += 20
    if bright > 0.25:
        scores["light"] += 15

    # Dệt may: ít kim loại nhưng texture mềm, không quá đồng nhất
    if dark < 0.10 and 30 <= std <= 55:
        scores["textile"] += 30
    if 0.035 <= edge <= 0.08:
        scores["textile"] += 20

    # Điện tử/pin: nhiều cụm đậm nhỏ, cạnh tương đối cao
    if very_dark > 0.03:
        scores["electronics"] += 25
    if regions >= 4:
        scores["electronics"] += 25
    if edge > 0.06:
        scores["electronics"] += 15

    # Hàng hỗn hợp
    if edge > 0.075 and std > 45 and regions >= 3:
        scores["mixed"] += 45
    if dark > 0.18 and bright > 0.15:
        scores["mixed"] += 20
    if lap > 350:
        scores["mixed"] += 15

    predicted = max(scores, key=scores.get)
    confidence = scores[predicted]

    if confidence < 35:
        predicted = "unknown"

    return predicted, scores


def compare_manifest(manifest_groups, predicted_type, scores, a):
    risks = []
    risk_score = 0

    edge = a["edge_ratio"]
    std = a["std_density"]
    dark = a["dark_ratio"]
    very_dark = a["very_dark_ratio"]
    regions = a["suspicious_regions"]
    large_regions = a["large_dark_regions"]

    if "unknown" in manifest_groups:
        risks.append("Manifest chưa xác định rõ nhóm hàng, cần nhập mô tả cụ thể hơn.")
        risk_score += 20

    if predicted_type != "unknown" and predicted_type not in manifest_groups:
        risks.append(
            f"AI suy đoán ảnh giống nhóm '{predicted_type}' nhưng Manifest khai {manifest_groups}."
        )
        risk_score += 55

    if "brick" in manifest_groups:
        if predicted_type in ["machinery", "electronics", "mixed"]:
            risks.append("Manifest khai gạch/vật liệu xây dựng nhưng ảnh có đặc trưng không đồng nhất, giống máy móc/hàng hỗn hợp.")
            risk_score += 35
        if edge > 0.055:
            risks.append("Ảnh có nhiều cạnh/cấu trúc hơn mức thường gặp của hàng gạch xếp đều.")
            risk_score += 20
        if dark > 0.12:
            risks.append("Ảnh có vùng đậm đáng kể, không phù hợp với khai báo gạch thông thường.")
            risk_score += 20
        if regions >= 2:
            risks.append("Có nhiều vùng đậm được khoanh, cần kiểm tra khả năng hàng khác lẫn trong container.")
            risk_score += 20

    if "light" in manifest_groups or "textile" in manifest_groups:
        if dark > 0.10:
            risks.append("Manifest khai hàng nhẹ/hàng mềm nhưng ảnh có tỷ lệ vùng đậm cao.")
            risk_score += 25
        if predicted_type in ["machinery", "electronics", "mixed"]:
            risks.append("Ảnh có đặc trưng giống hàng đặc/máy móc, không phù hợp khai báo hàng nhẹ.")
            risk_score += 35

    if "machinery" not in manifest_groups:
        if predicted_type == "machinery":
            risks.append("Manifest không khai máy móc nhưng ảnh có đặc trưng máy móc/kim loại.")
            risk_score += 35

    if "electronics" not in manifest_groups:
        if very_dark > 0.035 and regions >= 3:
            risks.append("Có nhiều cụm đậm nhỏ, cần lưu ý khả năng linh kiện/pin/hàng đặc không khai báo.")
            risk_score += 25

    if large_regions >= 2:
        risks.append("Có nhiều vùng đậm lớn, cần kiểm tra thủ công.")
        risk_score += 15

    if not risks:
        risks.append("Chưa phát hiện mâu thuẫn rõ, nhưng vẫn cần cán bộ kiểm tra trực quan.")
        risk_score = 10

    risk_score = min(risk_score, 100)

    if risk_score >= 70:
        level = "RỦI RO CAO"
    elif risk_score >= 40:
        level = "RỦI RO TRUNG BÌNH"
    else:
        level = "RỦI RO THẤP"

    return risk_score, level, risks


if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    manifest_groups = classify_manifest(manifest)
    analysis = analyze_image(img)
    predicted_type, cargo_scores = predict_cargo_type(analysis)
    risk_score, level, risks = compare_manifest(
        manifest_groups,
        predicted_type,
        cargo_scores,
        analysis
    )

    st.subheader("1. Ảnh gốc")
    st.image(img, width="stretch")

    st.subheader("2. Ảnh khoanh vùng nghi vấn")
    st.image(analysis["result"], width="stretch")

    st.subheader("3. Chỉ số kỹ thuật ảnh X-ray")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Chỉ số cạnh/cấu trúc", f"{analysis['edge_ratio']:.3f}")
        st.metric("Độ biến thiên mật độ", f"{analysis['std_density']:.1f}")
        st.metric("Tỷ lệ vùng đậm", f"{analysis['dark_ratio']:.2%}")
        st.metric("Tỷ lệ vùng rất đậm", f"{analysis['very_dark_ratio']:.2%}")

    with col2:
        st.metric("Mật độ trung bình", f"{analysis['mean_density']:.1f}")
        st.metric("Vùng sáng/rỗng", f"{analysis['bright_ratio']:.2%}")
        st.metric("Số vùng nghi vấn", analysis["suspicious_regions"])
        st.metric("Vùng đậm lớn", analysis["large_dark_regions"])

    st.subheader("4. AI suy đoán nhóm hàng từ ảnh")
    st.write(f"### Nhóm hàng ảnh giống nhất: `{predicted_type}`")

    st.write("Điểm nhận dạng từng nhóm:")
    st.json(cargo_scores)

    st.subheader("5. Nhóm hàng theo Manifest")
    st.write(manifest_groups)

    st.subheader("6. Đối chiếu Manifest")
    st.metric("Điểm rủi ro", f"{risk_score}/100")
    st.write(f"### Mức đánh giá: {level}")

    if risk_score >= 70:
        st.error("Rủi ro cao. Nên kiểm tra thủ công/soi chiếu bổ sung.")
    elif risk_score >= 40:
        st.warning("Có dấu hiệu cần lưu ý. Nên kiểm tra thêm.")
    else:
        st.success("Rủi ro thấp theo các chỉ số hiện tại.")

    st.subheader("7. Lý do đánh giá")
    for r in risks:
        st.write(f"- {r}")

    st.subheader("8. Ảnh tăng tương phản")
    st.image(analysis["enhanced"], width="stretch", clamp=True)

    st.subheader("9. Ảnh cạnh/cấu trúc")
    st.image(analysis["edges"], width="stretch", clamp=True)

else:
    st.info("Nhập Manifest và tải ảnh X-ray container để bắt đầu phân tích.")