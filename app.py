import base64, io, os, traceback
import cv2, numpy as np, requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title='Container X-Ray AI V11 Backend')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

MODELS=['gemini-3-flash-preview','gemini-2.5-flash','gemini-2.5-flash-lite','gemini-2.0-flash','gemini-1.5-flash','gemini-1.5-flash-002']

def key(): return os.getenv('GEMINI_API_KEY','').strip()

def load_img(b):
    pil=Image.open(io.BytesIO(b)).convert('RGB')
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def data_url(img):
    ok,buf=cv2.imencode('.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),90])
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.tobytes()).decode()

def b64(img):
    ok,buf=cv2.imencode('.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),90])
    return base64.b64encode(buf.tobytes()).decode()

def preprocess(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.8,tileGridSize=(8,8))
    enh=clahe.apply(gray)
    enh=cv2.fastNlMeansDenoising(enh,None,8,7,21)
    blur=cv2.GaussianBlur(enh,(0,0),1.2)
    sharp=cv2.addWeighted(enh,1.6,blur,-0.6,0)
    heat=cv2.applyColorMap(255-sharp,cv2.COLORMAP_JET)
    heat=cv2.addWeighted(img,0.62,heat,0.38,0)
    _,mask=cv2.threshold(sharp,88,255,cv2.THRESH_BINARY_INV)
    k=np.ones((5,5),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k,iterations=1)
    return sharp,heat,mask

def detect(mask,enh):
    h,w=mask.shape[:2]; area_img=w*h
    cont,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for c in cont:
        area=cv2.contourArea(c)
        if area<area_img*.002 or area>area_img*.55: continue
        x,y,bw,bh=cv2.boundingRect(c)
        aspect=bw/max(bh,1); fill=area/max(bw*bh,1)
        roi=enh[y:y+bh,x:x+bw]; density=int(255-np.mean(roi)) if roi.size else 0
        if aspect>3: label='vật thể dài / khung / thanh'
        elif aspect<.35: label='vật thể đứng / kiện cao'
        elif fill>.55: label='khối đặc / kiện hàng'
        elif fill<.25: label='cụm không đồng nhất / khoang rỗng'
        else: label='cụm vật thể mật độ cao'
        cx=x+bw/2; cy=y+bh/2
        zx='trái' if cx<w/3 else 'giữa' if cx<2*w/3 else 'phải'
        zy='trên' if cy<h/3 else 'giữa' if cy<2*h/3 else 'dưới'
        out.append({'label':label,'box':[int(x),int(y),int(bw),int(bh)],'area_percent':round(area/area_img*100,2),'density':density,'zone':f'{zy}-{zx}'})
    return sorted(out,key=lambda d:d['area_percent'],reverse=True)[:15]

def overlay(img,dets):
    out=img.copy()
    for i,d in enumerate(dets):
        x,y,w,h=d['box']; color=(0,60,255) if i<3 else (0,220,255)
        cv2.rectangle(out,(x,y),(x+w,y+h),color,3)
        txt=f"{i+1}. {d['label']}"
        cv2.rectangle(out,(x,max(0,y-30)),(min(out.shape[1],x+430),y),color,-1)
        cv2.putText(out,txt,(x+6,max(20,y-9)),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,0,0),2,cv2.LINE_AA)
    return out

def risk(mask,dets):
    h,w=mask.shape[:2]
    dark=np.count_nonzero(mask)/(w*h)*100
    return int(min(100, min(35,len(dets)*5)+min(45,dark*1.25)+min(20,sum(1 for d in dets if d['area_percent']>8)*7)))

def list_models(k):
    r=requests.get(f'https://generativelanguage.googleapis.com/v1beta/models?key={k}',timeout=25)
    d=r.json()
    if 'error' in d: raise RuntimeError(d['error'].get('message','Gemini API error'))
    return [m['name'].replace('models/','') for m in d.get('models',[]) if 'generateContent' in m.get('supportedGenerationMethods',[])]

def gemini(k,model,img64,manifest,dets,score):
    det='\n'.join([f"{i+1}. {x['label']}, vị trí {x['zone']}, diện tích {x['area_percent']}%, mật độ {x['density']}" for i,x in enumerate(dets)]) or 'Chưa phát hiện vùng rõ.'
    prompt=f'''Bạn là chuyên gia phân tích ảnh soi chiếu container hỗ trợ cán bộ Hải quan Việt Nam.

MANIFEST:
{manifest or 'Chưa nhập manifest.'}

DỮ LIỆU BACKEND:
Risk score kỹ thuật: {score}/100
Số vùng phát hiện: {len(dets)}
Danh sách vùng:
{det}

YÊU CẦU:
1. Mô tả vật thể chính trên ảnh đã khoanh vùng.
2. Phân tích mật độ, vùng đậm, vùng rỗng, cụm không đồng nhất.
3. Đối chiếu manifest.
4. Nêu dấu hiệu nghi vấn nghiệp vụ nếu có.
5. Chấm điểm rủi ro tổng hợp 0-100.
6. Kết luận: THÔNG QUAN / CẦN ĐỐI CHIẾU THÊM / GIỮ KIỂM TRA.
7. Đưa checklist kiểm tra thực tế nếu cần.

Trả lời tiếng Việt, thực tế, không khẳng định quá mức nếu ảnh không đủ rõ.'''
    payload={'contents':[{'parts':[{'inline_data':{'mime_type':'image/jpeg','data':img64}},{'text':prompt}]}],'generationConfig':{'temperature':0.2,'maxOutputTokens':2400}}
    r=requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={k}',json=payload,timeout=70)
    d=r.json()
    if 'error' in d: raise RuntimeError(f"[{model}] {d['error'].get('message','Gemini error')}")
    return d.get('candidates',[{}])[0].get('content',{}).get('parts',[{}])[0].get('text','')

@app.get('/')
def root(): return {'ok':True,'service':'Container X-Ray AI V11 Backend'}

@app.get('/health')
def health():
    k=key()
    return {'ok':True,'service':'Container X-Ray AI V11 Backend','gemini_key_configured':bool(k),'key_prefix':(k[:2]+'***') if k else None}

@app.post('/analyze')
async def analyze(image:UploadFile=File(...), manifest:str=Form('')):
    try:
        k=key()
        if not k: return JSONResponse(status_code=500,content={'ok':False,'error':'Backend chưa cấu hình GEMINI_API_KEY trên Render.'})
        img=load_img(await image.read())
        h,w=img.shape[:2]
        if w>1400:
            img=cv2.resize(img,(1400,int(h*1400/w)))
        enh,heat,mask=preprocess(img)
        dets=detect(mask,enh)
        ov=overlay(img,dets)
        score=risk(mask,dets)
        models=list_models(k)
        candidates=[m for m in MODELS if m in models] or models
        ai=''; used=''; last=''
        ov64=b64(ov)
        for m in candidates:
            try:
                ai=gemini(k,m,ov64,manifest,dets,score)
                if ai: used=m; break
            except Exception as e: last=str(e)
        if not ai: ai='Không gọi được Gemini. Lỗi cuối: '+last
        low=ai.lower()
        conclusion='CẦN ĐỐI CHIẾU THÊM'
        if score>=70 or 'giữ kiểm tra' in low or 'không phù hợp' in low: conclusion='GIỮ KIỂM TRA'
        elif score<=35 and ('thông quan' in low or 'phù hợp' in low): conclusion='CÓ THỂ THÔNG QUAN NẾU HỒ SƠ PHÙ HỢP'
        return {'ok':True,'risk_score':score,'conclusion':conclusion,'model_used':used,'ai_text':ai,'detections':dets,'images':{'overlay':data_url(ov),'enhanced':data_url(cv2.cvtColor(enh,cv2.COLOR_GRAY2BGR)),'heat':data_url(heat)}}
    except Exception as e:
        return JSONResponse(status_code=500,content={'ok':False,'error':str(e),'trace':traceback.format_exc()})
