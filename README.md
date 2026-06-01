Container X-Ray AI V11
Cách dùng
`index.html`: đưa lên GitHub Pages.
`backend/app.py`: đưa lên Render.
Gemini API key cài trong Render bằng biến môi trường `GEMINI_API_KEY`.
Render
Root Directory: `backend`
Build Command: `pip install -r requirements.txt`
Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
Environment Variable:
GEMINI_API_KEY = key Gemini của bạn
Frontend
Mở app GitHub Pages, nhập Backend URL của Render, bấm kiểm tra backend rồi phân tích ảnh.
