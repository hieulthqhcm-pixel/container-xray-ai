import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI phân tích X-ray container + Manifest",
    layout="wide"
)

st.title("AI phân tích X-ray container và đối chiếu Manifest")
st.caption("Bản hỗ trợ nghiệp vụ: phân tích mật độ, cấu trúc, độ đồng nhất và so sánh khai báo hàng hóa.")

manifest = st.text_area(
    "Nhập Manifest / khai báo hàng hóa",
    placeholder="Ví dụ: ceramic tiles, bricks, machinery, plastic toys, textile, furniture..."
)

uploaded = st.file_uploader(
    "Tải ảnh X-ray container",
    type=["jpg", "jpeg", "png", "bmp"]
)

def keyword_group(text):
    text = text.lower()

    groups = {
        "Gạch / đá / vật liệu xây dựng": [
            "brick", "bricks", "gạch", "ceramic", "tile", "tiles",
            "stone", "marble", "granite", "cement"
        ],
        "Máy móc / thiết bị / kim loại": [
            "machine", "machinery", "equipment", "engine", "motor",
            "forklift", "steel", "iron", "metal", "tool", "parts"
        ],
        "Dệt may / hàng mềm": [
            "clothes", "garment", "textile", "fabric", "shoes",
            "bag", "bags", "cotton"
        ],
        "Nhựa / đồ chơi / hàng nhẹ": [
            "plastic", "toy", "toys", "paper", "carton", "foam",
            "polystyrene"
        ],
        "Điện tử / pin / linh kiện": [
            "battery", "lithium", "electronics", "computer", "phone",
            "circuit", "adapter", "charger"
        ],
        "Nội thất / gỗ": [
            "furniture", "wood", "chair", "table", "cabinet", "sofa"
        ],
        "Thực phẩm / hữu cơ": [
            "food", "fruit", "vegetable", "seafood", "meat", "organic"
        ]
    }

    matched = []

    for group, words in groups.items():
        if any(w in text for w in words):
            matched.append(group)

    if not matched:
        matched.append("Không xác định rõ nhóm hàng")

    return matched


def analyze_xray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Tăng tương phản
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Cạnh ảnh
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edge_ratio = np.count_nonzero(edges) / edges.size

    # Mật độ và độ biến thiên
    mean_density = float(np.mean(gray))
    std_density = float(np.std(gray))

    # Vùng đậm
    dark_mask = gray < 70
    dark_ratio = float(np.count_nonzero(dark_mask) / dark_mask.size)

    # Vùng sáng/rỗng
    bright_mask = gray > 210
    bright_ratio = float(np.count_nonzero(bright_mask) / bright_mask.size)

    # Tìm contour vùng đậm nghi vấn
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(dark_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    suspicious_regions = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            suspicious_regions += 1
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(
                result,
                "NGHI VAN",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    # Phân loại đặc trưng ảnh
    image_features = []

    if edge_ratio > 0.09:
        image_features.append("Cấu trúc phức tạp, nhiều đường biên - giống máy móc/thiết bị/hàng hỗn hợp")

    if std_density > 55:
        image_features.append("Mật độ biến thiên cao - hàng không đồng nhất")

    if dark_ratio > 0.18:
        image_features.append("Nhiều vùng đậm - nghi kim loại, máy móc hoặc vật thể đặc")

    if bright_ratio > 0.35:
        image_features.append("Nhiều vùng rỗng/sáng - cần kiểm tra khoảng trống bất thường")

    if edge_ratio < 0.045 and std_density < 38:
        image_features.append("Cấu trúc tương đối đồng đều - gần với gạch, vật liệu xây dựng, hàng xếp đều")

    if not image_features:
        image_features.append("Ảnh chưa có dấu hiệu đặc biệt rõ")

    return {
        "gray": gray,
        "enhanced": enhanced,
        "edges": edges,
        "result": result,
        "edge_ratio": edge_ratio,
        "mean_density": mean_density,
        "std_density": std_density,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "suspicious_regions": suspicious_regions,
        "features": image_features
    }


def compare_manifest(manifest_text, groups, analysis):
    risks = []
    score = 0

    text = manifest_text.lower()

    declared_brick = "Gạch / đá / vật liệu xây dựng" in groups
    declared_machine = "Máy móc / thiết bị / kim loại" in groups
    declared_soft = "Dệt may / hàng mềm" in groups
    declared_light = "Nhựa / đồ chơi / hàng nhẹ" in groups
    declared_electronic = "Điện tử / pin / linh kiện" in groups
    declared_wood = "Nội thất / gỗ" in groups

    edge = analysis["edge_ratio"]
    std = analysis["std_density"]
    dark = analysis["dark_ratio"]
    bright = analysis["bright_ratio"]
    regions = analysis["suspicious_regions"]

    if manifest_text.strip() == "":
        risks.append("Chưa nhập Manifest nên chưa thể đối chiếu khai báo.")
        score += 20

    if declared_brick:
        if edge > 0.075 or std > 50:
            risks.append("Manifest khai gạch/vật liệu xây dựng nhưng ảnh có cấu trúc phức tạp, không giống hàng xếp đều.")
            score += 35
        if dark > 0.20:
            risks.append("Manifest khai gạch nhưng ảnh có nhiều vùng đậm, cần kiểm tra khả năng lẫn kim loại/máy móc.")
            score += 25

    if declared_soft or declared_light:
        if dark > 0.15:
            risks.append("Manifest khai hàng nhẹ/hàng mềm nhưng ảnh có nhiều vùng đậm, nghi hàng đặc hoặc kim loại.")
            score += 30
        if edge > 0.08:
            risks.append("Manifest khai hàng nhẹ/hàng mềm nhưng ảnh có nhiều cấu trúc cạnh phức tạp.")
            score += 25

    if not declared_machine:
        if edge > 0.11 and std > 55:
            risks.append("Manifest không khai máy móc nhưng ảnh có đặc trưng giống máy móc/thiết bị/hàng hỗn hợp.")
            score += 35

    if not declared_electronic:
        if dark > 0.22 and regions >= 3:
            risks.append("Ảnh có nhiều khối đậm lặp lại, cần kiểm tra khả năng linh kiện/pin/hàng đặc không khai báo.")
            score += 25

    if declared_machine:
        if edge < 0.04 and std < 35:
            risks.append("Manifest khai máy móc nhưng ảnh khá đồng đều, cần đối chiếu lại mô tả hàng.")
            score += 20

    if bright > 0.45:
        risks.append("Ảnh có nhiều khoảng sáng/rỗng, cần kiểm tra khả năng thiếu hàng hoặc bố trí bất thường.")
        score += 15

    if regions >= 5:
        risks.append("Có nhiều vùng đậm nghi vấn được khoanh vùng tự động.")
        score += 20

    if not risks:
        risks.append("Chưa phát hiện mâu thuẫn rõ, nhưng vẫn cần cán bộ kiểm tra trực quan.")
        score += 5

    score = min(score, 100)

    if score >= 70:
        level = "RỦI RO CAO"
    elif score >= 40:
        level = "RỦI RO TRUNG BÌNH"
    else:
        level = "RỦI RO THẤP"

    return score, level, risks


if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    groups = keyword_group(manifest)
    analysis = analyze_xray(img)
    score, level, risks = compare_manifest(manifest, groups, analysis)

    st.subheader("1. Ảnh gốc")
    st.image(img, width="stretch")

    st.subheader("2. Ảnh khoanh vùng nghi vấn")
    st.image(analysis["result"], width="stretch")

    st.subheader("3. Kết quả phân tích ảnh X-ray")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Chỉ số cạnh ảnh", f"{analysis['edge_ratio']:.3f}")
        st.metric("Độ biến thiên mật độ", f"{analysis['std_density']:.1f}")
        st.metric("Tỷ lệ vùng đậm", f"{analysis['dark_ratio']:.2%}")

    with col2:
        st.metric("Mật độ trung bình", f"{analysis['mean_density']:.1f}")
        st.metric("Tỷ lệ vùng sáng/rỗng", f"{analysis['bright_ratio']:.2%}")
        st.metric("Vùng đậm nghi vấn", analysis["suspicious_regions"])

    st.subheader("4. Đặc trưng ảnh AI nhận thấy")
    for f in analysis["features"]:
        st.info(f)

    st.subheader("5. Nhóm hàng theo Manifest")
    for g in groups:
        st.write(f"- {g}")

    st.subheader("6. Đối chiếu Manifest")

    st.metric("Điểm rủi ro", f"{score}/100")
    st.write(f"### Mức đánh giá: {level}")

    if score >= 70:
        st.error("Cảnh báo rủi ro cao. Nên kiểm tra thủ công/soi chiếu bổ sung.")
    elif score >= 40:
        st.warning("Có dấu hiệu cần lưu ý. Nên kiểm tra thêm.")
    else:
        st.success("Rủi ro thấp theo các chỉ số hiện tại.")

    st.subheader("7. Lý do đánh giá")
    for r in risks:
        st.write(f"- {r}")

    st.subheader("8. Ảnh tăng tương phản")
    st.image(analysis["enhanced"], width="stretch", clamp=True)

    st.subheader("9. Ảnh cạnh cấu trúc")
    st.image(analysis["edges"], width="stretch", clamp=True)

else:
    st.info("Vui lòng nhập Manifest và tải ảnh X-ray container để bắt đầu phân tích.")