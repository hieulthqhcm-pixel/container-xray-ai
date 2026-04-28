import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="AI X-ray container + Manifest", layout="wide")

st.title("AI phân tích X-ray container và so sánh Manifest")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

manifest = st.text_area(
    "Nhập Manifest / khai báo hàng hóa",
    placeholder="Ví dụ: 100 cartons of clothes, plastic accessories, no metal machinery..."
)

uploaded = st.file_uploader("Tải ảnh X-ray container", type=["jpg", "jpeg", "png", "bmp"])

conf = st.slider("Độ tin cậy AI", 0.1, 0.9, 0.25, 0.05)

def basic_manifest_check(manifest_text, detections):
    text = manifest_text.lower()
    risk_notes = []

    metal_words = [
        "metal", "steel", "iron", "machine", "machinery",
        "engine", "motor", "forklift", "tool", "equipment"
    ]

    soft_goods = [
        "clothes", "garment", "textile", "fabric",
        "shoes", "plastic", "toys", "paper", "carton"
    ]

    has_metal_declared = any(w in text for w in metal_words)
    has_soft_declared = any(w in text for w in soft_goods)

    if manifest_text.strip() == "":
        risk_notes.append("Chưa nhập Manifest nên chưa thể đối chiếu khai báo.")

    if has_soft_declared and len(detections) >= 5:
        risk_notes.append("Manifest khai hàng nhẹ/mềm nhưng ảnh có nhiều vùng/vật thể nghi vấn.")

    if not has_metal_declared and len(detections) >= 3:
        risk_notes.append("Manifest không khai máy móc/kim loại nhưng AI phát hiện nhiều vùng bất thường.")

    return risk_notes

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    st.subheader("Ảnh gốc")
    st.image(img, use_container_width=True)

    with st.spinner("AI đang phân tích ảnh..."):
        results = model.predict(img, conf=conf)

    annotated = results[0].plot()
    boxes = results[0].boxes

    detections = []

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        score = float(box.conf[0])
        detections.append({"label": label, "score": score})

    st.subheader("Kết quả AI")
    st.image(annotated, use_container_width=True)

    st.success(f"AI phát hiện {len(detections)} vùng/vật thể nghi vấn")

    if detections:
        st.subheader("Danh sách phát hiện")
        for i, d in enumerate(detections):
            st.write(f"{i+1}. {d['label']} - độ tin cậy: {d['score']:.2f}")

    st.subheader("Đối chiếu Manifest")

    risk_notes = basic_manifest_check(manifest, detections)

    if risk_notes:
        for note in risk_notes:
            st.warning(note)
    else:
        st.success("Chưa phát hiện dấu hiệu mâu thuẫn rõ với Manifest.")