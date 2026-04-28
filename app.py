import streamlit as st
from PIL import Image
import numpy as np
import cv2
import base64
from openai import OpenAI

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Container X-ray AI V4", layout="wide")

st.title("Container X-ray Intelligence V4 Professional")
st.caption("AI Vision phân tích ảnh X-ray container & đối chiếu Manifest")

# =========================
# OPENAI CLIENT
# =========================
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

# =========================
# INPUT
# =========================
manifest = st.text_area(
    "Nhập Manifest / khai báo hàng hóa",
    height=180,
    placeholder="Ví dụ:\nTOTAL:864 CARTONS CERAMIC FLOWERPOT HS CODE:691390"
)

uploaded_file = st.file_uploader(
    "Tải ảnh X-ray container",
    type=["jpg", "jpeg", "png"]
)

# =========================
# IMAGE TO BASE64
# =========================
def image_to_base64(img):
    import io
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# =========================
# AI ANALYSIS
# =========================
def analyze_with_ai(image, manifest_text):

    base64_image = image_to_base64(image)

    prompt = f"""
Bạn là chuyên gia soi chiếu container Hải quan.

NHIỆM VỤ:

1. Phân tích ảnh X-ray container thực tế.
2. Xác định:
- hàng đồng nhất hay không
- có cấu trúc máy móc/kim loại không
- mật độ hàng
- vùng bất thường
- khả năng che giấu
- khả năng sai khai báo Manifest

3. So sánh THỰC TẾ ảnh với manifest.

Manifest:
{manifest_text}

YÊU CẦU RẤT QUAN TRỌNG:

- KHÔNG đoán bừa.
- Chỉ kết luận theo hình ảnh thực tế.
- Nếu manifest khai gạch/ceramic nhưng ảnh KHÔNG giống gạch đồng nhất -> phải cảnh báo.
- Nếu ảnh có máy móc/kim loại/cấu trúc cơ khí -> phải nêu rõ.
- Nếu không đủ cơ sở -> nói "không đủ cơ sở kết luận".
- Không được copy kết quả cũ.
- Phải đánh giá đúng theo từng ảnh mới upload.

TRẢ KẾT QUẢ:
- Nhóm hàng ảnh giống nhất
- Mức độ phù hợp manifest
- Điểm rủi ro 0-100
- Nhận định chuyên sâu
- Khuyến nghị kiểm tra
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ]
    )

    return response.output_text

# =========================
# IMAGE ANALYSIS
# =========================
def image_metrics(image):

    img = np.array(image.convert("L"))

    mean = np.mean(img)
    std = np.std(img)

    dark_ratio = np.sum(img < 60) / img.size * 100
    very_dark_ratio = np.sum(img < 30) / img.size * 100
    bright_ratio = np.sum(img > 210) / img.size * 100

    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "dark_ratio": round(dark_ratio, 2),
        "very_dark_ratio": round(very_dark_ratio, 2),
        "bright_ratio": round(bright_ratio, 2),
        "edge_density": round(edge_density, 4)
    }

# =========================
# MAIN
# =========================
if uploaded_file:

    image = Image.open(uploaded_file)

    st.subheader("1. Ảnh gốc")
    st.image(image, use_container_width=True)

    # -------------------------
    # TECHNICAL ANALYSIS
    # -------------------------
    metrics = image_metrics(image)

    st.subheader("2. Chỉ số kỹ thuật ảnh")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Mật độ trung bình", metrics["mean"])
        st.metric("Độ biến thiên", metrics["std"])
        st.metric("Tỷ lệ vùng đậm", f'{metrics["dark_ratio"]}%')

    with col2:
        st.metric("Tỷ lệ vùng rất đậm", f'{metrics["very_dark_ratio"]}%')
        st.metric("Tỷ lệ vùng sáng/rỗng", f'{metrics["bright_ratio"]}%')
        st.metric("Mật độ cạnh/cấu trúc", metrics["edge_density"])

    # -------------------------
    # ENHANCED IMAGE
    # -------------------------
    st.subheader("3. Ảnh tăng tương phản")

    gray = np.array(image.convert("L"))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    st.image(enhanced, use_container_width=True)

    # -------------------------
    # AI ANALYSIS
    # -------------------------
    if manifest.strip():

        with st.spinner("AI đang phân tích ảnh X-ray..."):

            try:

                result = analyze_with_ai(image, manifest)

                st.subheader("4. Kết quả AI Vision")
                st.write(result)

            except Exception as e:

                st.error(str(e))

else:

    st.info("Vui lòng upload ảnh X-ray container.")