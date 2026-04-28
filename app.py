import streamlit as st
from openai import OpenAI
from PIL import Image
import base64
import json
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Container X-ray V4 Professional", layout="wide")

st.title("Container X-ray Risk Intelligence V4 Professional")
st.caption("AI Vision phân tích ảnh X-ray, đối chiếu Manifest và xuất nhận định rủi ro.")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

manifest = st.text_area(
    "Nhập Manifest / khai báo hàng hóa",
    placeholder="Ví dụ: TOTAL:864 CARTONS CERAMIC FLOWERPOT HS CODE:691390"
)

uploaded = st.file_uploader(
    "Tải ảnh X-ray container",
    type=["jpg", "jpeg", "png", "bmp"]
)

def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def analyze_with_ai(image, manifest_text):
    img_b64 = image_to_base64(image)

    prompt = f"""
Bạn là hệ thống hỗ trợ phân tích ảnh soi chiếu X-ray container cho mục đích đối chiếu Manifest.

Yêu cầu:
1. Quan sát ảnh X-ray.
2. Mô tả hàng hóa nhìn thấy trong ảnh.
3. Suy đoán nhóm hàng từ ảnh.
4. Đọc Manifest người dùng nhập.
5. So sánh ảnh với Manifest.
6. Chấm điểm rủi ro từ 0 đến 100.
7. Không kết luận chắc chắn hàng cấm. Chỉ nêu dấu hiệu nghi vấn cần kiểm tra.
8. Trả lời bằng JSON hợp lệ, không markdown.

Manifest:
{manifest_text}

Schema JSON:
{{
  "image_description": "...",
  "image_cargo_guess": ["..."],
  "manifest_goods": ["..."],
  "match_assessment": "...",
  "risk_score": 0,
  "risk_level": "LOW/MEDIUM/HIGH",
  "risk_reasons": ["..."],
  "recommended_actions": ["..."],
  "confidence": "LOW/MEDIUM/HIGH"
}}
"""

    response = client.responses.create(
        model="gpt-5.5",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{img_b64}"
                    }
                ]
            }
        ],
    )

    text = response.output_text.strip()

    try:
        return json.loads(text)
    except Exception:
        return {
            "image_description": text,
            "image_cargo_guess": [],
            "manifest_goods": [],
            "match_assessment": "Không parse được JSON.",
            "risk_score": 50,
            "risk_level": "MEDIUM",
            "risk_reasons": ["AI trả về kết quả không đúng định dạng JSON."],
            "recommended_actions": ["Kiểm tra thủ công và chạy lại ảnh rõ hơn."],
            "confidence": "LOW"
        }

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    st.subheader("1. Ảnh X-ray")
    st.image(image, width="stretch")

    if st.button("Phân tích bằng AI Vision"):
        with st.spinner("AI Vision đang phân tích ảnh và Manifest..."):
            result = analyze_with_ai(image, manifest)

        st.subheader("2. Mô tả ảnh")
        st.write(result.get("image_description", ""))

        st.subheader("3. Nhóm hàng AI suy đoán từ ảnh")
        st.write(result.get("image_cargo_guess", []))

        st.subheader("4. Nhóm hàng theo Manifest")
        st.write(result.get("manifest_goods", []))

        st.subheader("5. Đánh giá khớp Manifest")
        st.write(result.get("match_assessment", ""))

        st.subheader("6. Risk Score")
        score = int(result.get("risk_score", 0))
        level = result.get("risk_level", "MEDIUM")

        st.metric("Điểm rủi ro", f"{score}/100")

        if level == "HIGH":
            st.error("RỦI RO CAO")
        elif level == "MEDIUM":
            st.warning("RỦI RO TRUNG BÌNH")
        else:
            st.success("RỦI RO THẤP")

        st.subheader("7. Lý do đánh giá")
        for r in result.get("risk_reasons", []):
            st.write(f"- {r}")

        st.subheader("8. Khuyến nghị xử lý")
        for a in result.get("recommended_actions", []):
            st.write(f"- {a}")

        st.caption(f"Báo cáo tạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.info("Nhập Manifest và tải ảnh X-ray container để bắt đầu.")