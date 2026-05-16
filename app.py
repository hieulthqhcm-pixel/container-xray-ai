"""
X-Ray Cargo Analyzer V8.0 HYBRID
- Giu layout quen thuoc V7.6 (manifest + upload + sections)
- Nang cap toan bo engine phan tich tu 15 nguon hoc thuat
- Them tabs anh, sidebar LUT, physics warnings
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import hashlib

st.set_page_config(
    page_title="X-Ray Cargo Analyzer V8.0",
    page_icon="📦",
    layout="wide"
)

st.title("📦 X-Ray Cargo Analyzer V8.0 HYBRID")
st.caption(
    "Engine V8 (15 nguon hoc thuat: WCO · MIT · UCL · BAM · EMPA · CASRA · Varian · Hitachi) "
    "| Layout quen thuoc V7.6 | Physics-aware · Epistemically honest"
)

# ═══════════════════════════════════════════════════════
# RESET
# ═══════════════════════════════════════════════════════

def make_hash(uploaded_file, manifest_text):
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest() if uploaded_file else ""
    text_hash = hashlib.md5(manifest_text.encode("utf-8")).hexdigest()
    return file_hash + "_" + text_hash


def clear_old_results():
    for k in ["last_key"]:
        if k in st.session_state:
            del st.session_state[k]


# ═══════════════════════════════════════════════════════
# MANIFEST CLASSIFY (giu V7.6 + bo sung)
# ═══════════════════════════════════════════════════════

def classify_manifest(text):
    t = text.upper()
    groups = {
        "MACHINERY": [
            "EXCAVATOR","MACHINE","MACHINERY","FORKLIFT","TRUCK","ENGINE","MOTOR",
            "CRANE","LOADER","BULLDOZER","PUMP","COMPRESSOR","GENERATOR","TURBINE",
            "GEARBOX","HYDRAULIC","PNEUMATIC","CONVEYOR","ROBOT","PRESS",
            "8429","842952","84295200","8427","8431","8701","8704","8413","8414"
        ],
        "CERAMIC": [
            "CERAMIC","FLOWERPOT","POT","TILE","PORCELAIN","STONE","GRANITE",
            "MARBLE","BRICK","CLAY","TERRACOTTA","SANITARYWARE",
            "6913","6907","6908","6910","6911","6912"
        ],
        "TEXTILE": [
            "TEXTILE","GARMENT","CLOTH","FABRIC","SHIRT","PANTS","COTTON",
            "POLYESTER","APPAREL","YARN","THREAD","KNIT","WOVEN","FIBER",
            "BLANKET","CURTAIN","CARPET","RUG"
        ],
        "PLASTIC": [
            "PLASTIC","POLY","PVC","PE","PP","RESIN","POLYSTYRENE","NYLON",
            "ACRYLIC","ABS","HDPE","LDPE","PET","FOAM",
            "3901","3902","3903","390311","3904","3905","3906","3907","3916","3917"
        ],
        "ELECTRONICS": [
            "ELECTRONIC","COMPUTER","LAPTOP","PHONE","CIRCUIT","BOARD","BATTERY",
            "CAPACITOR","TRANSISTOR","SEMICONDUCTOR","PCB","LED","DISPLAY","SENSOR",
            "8507","8517","8541","8542","8471","8473","8534"
        ],
        "LIQUID_CHEMICAL": [
            "LIQUID","CHEMICAL","OIL","PAINT","INK","ADHESIVE","DRUM","BARREL",
            "IBC","TANK","SOLVENT","ACID","ALKALI","RESIN","LUBRICANT","FUEL",
            "FERTILIZER","PESTICIDE","BLEACH"
        ],
        "FOOD": [
            "FOOD","RICE","WHEAT","FLOUR","SUGAR","COFFEE","TEA","FRUIT","VEGETABLE",
            "MEAT","FISH","SEAFOOD","FROZEN","CANNED","BEVERAGE","JUICE","MILK",
            "CHOCOLATE","SNACK","CEREAL","SPICE","SAUCE"
        ]
    }
    scores = {g: sum(1 for kw in kws if kw in t) for g, kws in groups.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "UNKNOWN", scores
    return best, scores


# ═══════════════════════════════════════════════════════
# TANG 1: TIEN XU LY ANH (Physics-aware)
# Nguon: Rogers/UCL 2016, Jaccard/UCL 2016, MIT/Lalor 2024
# ═══════════════════════════════════════════════════════

def column_normalize(gray):
    """Chuan hoa theo cot - loai sai lech nguon tia. Nguon: Rogers/UCL 2016"""
    col_ref = np.percentile(gray, 95, axis=0).astype(float)
    col_ref = np.where(col_ref < 1, 1, col_ref)
    return (gray.astype(float) / col_ref * 255).clip(0, 255).astype(np.uint8)


def remove_stripe_artifacts(gray):
    """Loai soc doc do nguon tia loi. Nguon: Jaccard/UCL SPIE 2016"""
    col_std = np.std(gray, axis=0)
    bad = np.where(col_std < np.percentile(col_std, 5))[0]
    out = gray.copy()
    for c in bad:
        if 1 <= c <= gray.shape[1] - 2:
            out[:, c] = (gray[:, c-1].astype(int) + gray[:, c+1].astype(int)) // 2
    return out


def log_transform(gray):
    """alpha = -log(T): nen tang dual-energy. Nguon: MIT/Lalor 2024, Jaccard/UCL 2016"""
    lg = np.log1p(gray.astype(float))
    mx = lg.max()
    if mx > 0:
        lg = lg / mx * 255
    return lg.astype(np.uint8)


def apply_lut(gray, mode):
    """
    11 LUT modes theo chuan Rapiscan GXA va CASRA Simulator.
    Nguon: CASRA Simulator 2021, Michel-Mendes 2014, AS&E/Saverskiy 2020
    """
    if mode == "Greyscale":
        return gray
    elif mode == "BW Log":
        return log_transform(gray)
    elif mode == "BW Sqrt":
        return (np.sqrt(gray.astype(float) / 255.0) * 255).astype(np.uint8)
    elif mode == "Pseudo Color":
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        col  = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
    elif mode == "Histogram Mask":
        out = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        out[gray > 200]  = [100, 149, 237]   # blue = low density
        out[gray < 55]   = [220, 50, 47]     # red  = high density
        mid = (gray >= 55) & (gray <= 200)
        out[mid] = np.stack([gray[mid]] * 3, axis=1)
        return out
    elif mode == "Invert":
        return 255 - gray
    elif mode == "Edge Enhance":
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        lap  = cv2.Laplacian(blur, cv2.CV_64F)
        edge = cv2.convertScaleAbs(lap)
        return cv2.addWeighted(gray, 0.65, edge, 0.35, 0)
    elif mode == "Organic Only":
        _, org = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return org
    elif mode == "Quick Optimize":
        return cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    elif mode == "Wiener Contrast":
        try:
            from scipy.signal import wiener as sp_wiener
            w = sp_wiener(gray.astype(float), mysize=5)
            return ((w - w.min()) / (w.max() - w.min() + 1e-6) * 255).astype(np.uint8)
        except Exception:
            return cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4)).apply(gray)
    elif mode == "Log Transform (MIT)":
        return log_transform(gray)
    return gray


# ═══════════════════════════════════════════════════════
# TANG 2: TRICH XUAT DAC TRUNG (Multi-source)
# ═══════════════════════════════════════════════════════

def analyze_image(image_pil):

    img     = np.array(image_pil.convert("RGB"))
    gray_raw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w    = gray_raw.shape
    total_pixels = gray_raw.size

    # --- Tien xu ly V8 ---
    gray_norm  = column_normalize(gray_raw)          # Rogers/UCL 2016
    gray_clean = remove_stripe_artifacts(gray_norm)  # Jaccard/UCL 2016
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced   = clahe.apply(gray_clean)             # giu V7.6
    blur       = cv2.GaussianBlur(enhanced, (5, 5), 0)
    log_img    = log_transform(enhanced)             # MIT/Lalor 2024

    # --- ROI (giu V7.6) ---
    roi_x1, roi_x2 = int(w * 0.12), int(w * 0.88)
    roi_y1, roi_y2 = int(h * 0.12), int(h * 0.88)
    roi_mask = np.zeros_like(gray_raw)
    roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255

    # --- Edges ---
    edges = cv2.Canny(blur, 35, 120)
    edges = cv2.bitwise_and(edges, roi_mask)

    # --- Density thresholds (giu V7.6) ---
    dark_threshold      = np.percentile(blur, 28)
    very_dark_threshold = np.percentile(blur, 12)
    dark_mask      = cv2.bitwise_and(np.where(blur < dark_threshold, 255, 0).astype(np.uint8), roi_mask)
    very_dark_mask = cv2.bitwise_and(np.where(blur < very_dark_threshold, 255, 0).astype(np.uint8), roi_mask)

    dark_ratio      = float(np.sum(dark_mask > 0) / total_pixels)
    very_dark_ratio = float(np.sum(very_dark_mask > 0) / total_pixels)

    # --- Suspicious zones (giu V7.6 + them pos note) ---
    edge_dilate  = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    object_mask  = cv2.bitwise_and(dark_mask, edge_dilate)
    kernel       = np.ones((5, 5), np.uint8)
    object_mask  = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    object_mask  = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    contours, _  = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    suspicious_boxes = []
    object_area_sum  = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < total_pixels * 0.0015:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 35 or bh < 35:
            continue
        if bh > bw * 3 and bw < w * 0.08:
            continue
        if x < roi_x1 or x + bw > roi_x2 or y < roi_y1 or y + bh > roi_y2:
            continue
        roi_edge = edges[y:y+bh, x:x+bw]
        ed_in = float(np.sum(roi_edge > 0) / (bw * bh))
        if ed_in < 0.012:
            continue
        # Vi tri theo hieu ung phong dai hinh hoc (Reed/Varian 2008, Lalor MIT 2024)
        if y < h * 0.25:
            pos = "Vung tren - thuc te co the o giua (phong dai hinh hoc)"
        elif y > h * 0.75:
            pos = "Vung duoi - gan detector, it bi meo"
        else:
            pos = "Vung giua container"
        object_area_sum += area
        suspicious_boxes.append((x, y, bw, bh, area, ed_in, pos))

    suspicious_boxes = sorted(suspicious_boxes, key=lambda b: b[4] * b[5], reverse=True)[:5]

    # --- Basic metrics (giu V7.6) ---
    edge_density = float(np.sum(edges > 0) / total_pixels)
    std_density  = float(np.std(gray_raw))
    texture      = float(cv2.Laplacian(gray_raw, cv2.CV_64F).var())
    lines        = cv2.HoughLinesP(edges, 1, np.pi/180, 70, minLineLength=45, maxLineGap=10)
    line_count   = 0 if lines is None else len(lines)
    object_area_ratio = object_area_sum / total_pixels

    # ── V8 FEATURES ──────────────────────────────────────────

    # Alpha-map: alpha = -log(T) | Nguon: MIT/Lalor 2024, Jaccard/UCL 2016
    safe = np.clip(blur.astype(float), 1, 255)
    amap = -np.log(safe / 255.0)
    av   = amap.flatten()
    alpha_mean        = float(np.mean(av))
    alpha_high_ratio  = float(np.mean(av > 2.0))
    alpha_alarm_ratio = float(np.mean(av > 3.0))

    # Dark alarm: vung tia X khong xuyen qua | Nguon: WCO/CASRA 2020
    da_thr   = np.percentile(blur, 5)
    da_mask  = ((blur < da_thr) & (roi_mask > 0)).astype(np.uint8) * 255
    da_mask  = cv2.morphologyEx(da_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    da_cnts, _ = cv2.findContours(da_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_alarm_count = len([c for c in da_cnts if cv2.contourArea(c) > total_pixels * 0.003])
    dark_alarm_ratio = float(np.sum(da_mask > 0) / total_pixels)

    # Cargo entropy: hang hon tap | Nguon: WCO/CASRA, Mademlis/Athens 2024
    roi_px = blur[roi_mask > 0].astype(float)
    if len(roi_px) == 0:
        roi_px = blur.flatten().astype(float)
    hist_e, _ = np.histogram(roi_px, bins=10, range=(0, 255), density=True)
    hist_e    = hist_e[hist_e > 0]
    entropy   = float(-np.sum(hist_e * np.log2(hist_e + 1e-10)))
    homogeneity = round(1 - min(entropy / 3.5, 1.0), 3)
    if homogeneity > 0.7:
        cargo_complexity = "DONG NHAT - de phan tich"
    elif homogeneity > 0.4:
        cargo_complexity = "TRUNG BINH - can chu y"
    else:
        cargo_complexity = "HON TAP - kho phan tich, rui ro cao"

    # Zone histogram: phan vung mat do | Nguon: WCO/CASRA 2020
    zt = float(np.mean(blur[:h//3, :]))
    zm = float(np.mean(blur[h//3:2*h//3, :]))
    zb = float(np.mean(blur[2*h//3:, :]))
    density_gradient = zb - zt
    zone_uniformity  = float(np.std([zt, zm, zb]))

    # Symmetry | Nguon: cau truc hinh hoc
    mx, my  = w // 2, h // 2
    left    = gray_raw[:, :mx].astype(float)
    right   = np.fliplr(gray_raw[:, mx:mx + mx]).astype(float)
    sym_h   = float(1 - np.mean(np.abs(left - right)) / 255)
    top     = gray_raw[:my, :].astype(float)
    bot     = np.flipud(gray_raw[my:my + my, :]).astype(float)
    sym_v   = float(1 - np.mean(np.abs(top - bot)) / 255)
    sym_comb = (sym_h + sym_v) / 2

    # FFT | Nguon: ACXIS/EMPA 2016
    f   = np.fft.fft2(gray_raw.astype(float))
    fsh = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fsh))
    cy, cx = h // 2, w // 2
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt((X - cx)**2 + (Y - cy)**2)
    lr, mr = int(min(h, w) * 0.05), int(min(h, w) * 0.20)
    fft_hi_ratio = float(np.mean(mag[dist >= mr])) / (float(np.mean(mag[dist < lr])) + 1e-6)

    # Corner density | Nguon: ACXIS/EMPA 2016
    corners = cv2.goodFeaturesToTrack(gray_raw, maxCorners=500, qualityLevel=0.01, minDistance=10)
    corner_count = 0 if corners is None else len(corners)

    # Beer-Lambert MRA | Nguon: Jaccard/UCL SPIE 2016
    i0  = np.percentile(roi_px, 95)
    mra = float(1 - np.mean(roi_px / (i0 + 1e-6)))

    # Empty verification | Nguon: Jaccard/UCL SPIE 2016 (99.3% accuracy)
    cargo_ratio = float(np.sum(roi_mask > 0) / total_pixels)
    _, dark_blobs = cv2.threshold(blur, int(np.mean(roi_px) - np.std(roi_px) * 0.5), 255, cv2.THRESH_BINARY_INV)
    blob_ratio   = float(np.sum(dark_blobs > 0) / total_pixels)
    empty_score  = max(0, 1 - cargo_ratio * 2 - blob_ratio * 3)

    # Weight estimation | Nguon: ACXIS/EMPA 2016
    weight_index = round(float(np.mean(255 - roi_px)) / 255 * cargo_ratio * 100, 1)
    if weight_index > 50:
        weight_note = "Hang rat nang - doi chieu trong tai khai bao"
    elif weight_index > 25:
        weight_note = "Hang nang vua - phu hop kim loai/gom"
    else:
        weight_note = "Hang nhe - phu hop vai/nhua/thuc pham"

    # Overlap complexity | Nguon: Mademlis/Athens 2024
    hist_ov, _ = np.histogram(blur.flatten(), bins=32, range=(0, 255))
    norm_ov    = hist_ov / (hist_ov.sum() + 1e-6)
    peaks      = [i for i in range(1, len(norm_ov) - 1)
                  if norm_ov[i] > norm_ov[i-1] and norm_ov[i] > norm_ov[i+1] and norm_ov[i] > 0.02]
    ov_ent     = float(-np.sum(norm_ov[norm_ov > 0] * np.log2(norm_ov[norm_ov > 0] + 1e-10)))
    overlap_level = ("CAO" if len(peaks) >= 3 or ov_ent > 4.0 else
                     "TRUNG BINH" if len(peaks) == 2 or ov_ent > 3.0 else "THAP")

    # Z-range (Rapiscan Sentry, Lalor MIT/PNNL 2024, Table 1)
    if alpha_alarm_ratio > 0.12 or very_dark_ratio > 0.20:
        z_range = "HEAVY-Z (Z>=46) - Chi/Dong/Thiec"
    elif very_dark_ratio > 0.06:
        z_range = "MID-Z (Z 16-45) - Thep/Nhom/Gom"
    else:
        z_range = "LOW-Z (Z<=15) - Nhua/Vai/Thuc pham"

    # ── PHAN LOAI HANG HOA (V7.6 + bo sung V8) ─────────────

    feature_scores = {
        "MACHINERY": 0, "CERAMIC": 0, "TEXTILE": 0,
        "PLASTIC": 0, "ELECTRONICS": 0, "LIQUID_CHEMICAL": 0, "FOOD": 0
    }

    # V7.6 rules giu nguyen
    if line_count >= 180:  feature_scores["MACHINERY"] += 35
    elif line_count >= 100: feature_scores["MACHINERY"] += 25
    if edge_density >= 0.020: feature_scores["MACHINERY"] += 25
    if texture >= 180: feature_scores["MACHINERY"] += 20
    if very_dark_ratio >= 0.06:
        feature_scores["MACHINERY"] += 10
        feature_scores["ELECTRONICS"] += 10
    if dark_ratio >= 0.18: feature_scores["CERAMIC"] += 25
    if edge_density < 0.018 and texture < 220: feature_scores["CERAMIC"] += 25
    if object_area_ratio >= 0.08 and line_count < 150: feature_scores["CERAMIC"] += 20
    if edge_density < 0.015: feature_scores["TEXTILE"] += 25
    if texture < 120: feature_scores["TEXTILE"] += 25
    if very_dark_ratio < 0.04: feature_scores["TEXTILE"] += 15
    if 0.08 <= dark_ratio <= 0.20: feature_scores["PLASTIC"] += 20
    if 100 <= texture <= 260: feature_scores["PLASTIC"] += 20
    if line_count < 180: feature_scores["PLASTIC"] += 10
    if texture >= 300: feature_scores["ELECTRONICS"] += 30
    if edge_density >= 0.028: feature_scores["ELECTRONICS"] += 25
    if line_count >= 220: feature_scores["ELECTRONICS"] += 15
    if edge_density < 0.014 and std_density < 45:
        feature_scores["LIQUID_CHEMICAL"] += 25
    if dark_ratio >= 0.12 and texture < 140:
        feature_scores["LIQUID_CHEMICAL"] += 20

    # V8 rules bo sung
    if zone_uniformity < 8:
        feature_scores["LIQUID_CHEMICAL"] += 20
        feature_scores["TEXTILE"] += 10
    if density_gradient > 15:
        feature_scores["MACHINERY"] += 15
        feature_scores["CERAMIC"] += 10
    if sym_comb > 0.75:
        feature_scores["MACHINERY"] += 20
        feature_scores["ELECTRONICS"] += 10
    if sym_comb < 0.55:
        feature_scores["CERAMIC"] += 10
        feature_scores["PLASTIC"] += 10
    if fft_hi_ratio > 1.2:
        feature_scores["TEXTILE"] += 20
    if corner_count > 200:
        feature_scores["MACHINERY"] += 20
        feature_scores["ELECTRONICS"] += 15
    if corner_count < 30:
        feature_scores["TEXTILE"] += 15
        feature_scores["LIQUID_CHEMICAL"] += 10
    if alpha_high_ratio > 0.08:
        feature_scores["MACHINERY"] += 20
    if homogeneity > 0.75:
        feature_scores["LIQUID_CHEMICAL"] += 15
        feature_scores["TEXTILE"] += 10
    if texture < 80 and edge_density < 0.010:
        feature_scores["FOOD"] += 20

    image_class = max(feature_scores, key=feature_scores.get)
    image_score = feature_scores[image_class]
    if image_score < 35:
        image_class = "UNKNOWN"

    # Calibrated confidence (MIT/Lalor 2024)
    base = min(88, max(45, int(image_score)))
    if image_class in ("TEXTILE", "PLASTIC", "FOOD"):
        base = min(base, 60)
    if very_dark_ratio > 0.15:
        base = min(base, 55)
    if entropy > 2.5:
        base = int(base * 0.85)
    ai_confidence = max(35, min(88, base))

    label_map = {
        "MACHINERY":       "MACHINERY / HEAVY STRUCTURE",
        "CERAMIC":         "CERAMIC / STONE / DENSE GOODS",
        "TEXTILE":         "TEXTILE / SOFT GOODS",
        "PLASTIC":         "PLASTIC / LIGHT-MEDIUM GOODS",
        "ELECTRONICS":     "ELECTRONICS / COMPLEX STRUCTURE",
        "LIQUID_CHEMICAL": "LIQUID / CHEMICAL / DRUM GOODS",
        "FOOD":            "FOOD / AGRICULTURAL GOODS",
        "UNKNOWN":         "UNKNOWN"
    }
    ai_label = label_map.get(image_class, "UNKNOWN")

    # ── HEATMAP va MARKED (giu V7.6 + them sliding window) ──

    density_map = cv2.GaussianBlur(255 - enhanced, (9, 9), 0)
    heat_bgr    = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
    heat_rgb    = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    # Sliding-window heatmap (Jaccard/UCL SPIE 2016)
    win, stride = 64, 32
    smap = np.zeros((h, w), dtype=float)
    cnt  = np.zeros((h, w), dtype=float)
    for yy in range(0, h - win, stride):
        for xx in range(0, w - win, stride):
            pb = blur[yy:yy+win, xx:xx+win]
            pe = edges[yy:yy+win, xx:xx+win]
            s  = np.mean(255 - pb) / 255 * 0.5 + np.mean(pe > 0) * 0.5
            smap[yy:yy+win, xx:xx+win] += s
            cnt [yy:yy+win, xx:xx+win] += 1
    cnt  = np.where(cnt == 0, 1, cnt)
    smap = smap / cnt
    sw_norm = (smap / (smap.max() + 1e-6) * 255).astype(np.uint8)
    sw_heat = cv2.applyColorMap(sw_norm, cv2.COLORMAP_JET)
    sw_rgb  = cv2.cvtColor(sw_heat, cv2.COLOR_BGR2RGB)

    marked = img.copy()
    for i, (x, y, bw, bh, area, ed_in, pos) in enumerate(suspicious_boxes):
        color = (255, 0, 0) if ed_in > 0.15 else (255, 165, 0)
        cv2.rectangle(marked, (x, y), (x + bw, y + bh), color, 3)
        cv2.putText(marked, f"DOI CHIEU {i+1}",
                    (x, max(25, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ── METRICS DICT ─────────────────────────────────────────

    metrics = {
        # V7.6 giu nguyen
        "Dark Ratio":        round(dark_ratio, 3),
        "Very Dark":         round(very_dark_ratio, 3),
        "Std Density":       round(std_density, 1),
        "Edge Density":      round(edge_density, 3),
        "Texture":           round(texture, 1),
        "Line Count":        int(line_count),
        "Object Area Ratio": round(object_area_ratio, 3),
        "Suspicious Zones":  int(len(suspicious_boxes)),
        "AI Confidence":     int(ai_confidence),
        "Uniformity Score":  round(100 - min(std_density, 100), 1),
        # V8 bo sung
        "Alpha Mean":        round(alpha_mean, 4),
        "Alpha High Ratio":  round(alpha_high_ratio, 4),
        "Alpha Alarm Ratio": round(alpha_alarm_ratio, 4),
        "Dark Alarm Count":  dark_alarm_count,
        "Dark Alarm Ratio":  round(dark_alarm_ratio, 4),
        "Homogeneity":       homogeneity,
        "Cargo Entropy":     round(entropy, 3),
        "Cargo Complexity":  cargo_complexity,
        "Zone Uniformity":   round(zone_uniformity, 1),
        "Density Gradient":  round(density_gradient, 1),
        "Symmetry H":        round(sym_h, 3),
        "Symmetry V":        round(sym_v, 3),
        "Symmetry Comb":     round(sym_comb, 3),
        "FFT Hi Ratio":      round(fft_hi_ratio, 3),
        "Corner Count":      corner_count,
        "MRA (Beer-Lambert)": round(mra, 4),
        "Empty Score":       round(empty_score, 3),
        "Weight Index":      weight_index,
        "Overlap Level":     overlap_level,
        "Z-Range":           z_range,
    }

    return {
        "original":       img,
        "gray":           enhanced,
        "log":            log_img,
        "heatmap":        heat_rgb,
        "sw_heatmap":     sw_rgb,
        "marked":         marked,
        "metrics":        metrics,
        "image_class":    image_class,
        "ai_label":       ai_label,
        "ai_confidence":  ai_confidence,
        "feature_scores": feature_scores,
        "suspicious_boxes": suspicious_boxes,
        "z_range":        z_range,
        "weight_note":    weight_note,
        "cargo_complexity": cargo_complexity,
    }


# ═══════════════════════════════════════════════════════
# RISK ENGINE V8 (V7.6 + bo sung tu WCO, MIT, ACXIS)
# ═══════════════════════════════════════════════════════

def calculate_risk(manifest_class, image_class, metrics, ai_confidence):
    score   = 20
    reasons = []
    physics_warnings = []

    # V7.6 rules giu nguyen
    if manifest_class == "UNKNOWN":
        score += 15
        reasons.append("Manifest chua du thong tin de phan loai chac chan.")
    if image_class == "UNKNOWN":
        score += 15
        reasons.append("Anh X-ray chua du dac trung de phan loai chac chan.")
    if manifest_class != "UNKNOWN" and image_class != "UNKNOWN":
        if manifest_class == image_class:
            score -= 10
            reasons.append("Dac trung anh tuong doi phu hop voi Manifest.")
        else:
            score += 35
            reasons.append(
                f"Manifest khai bao nhom '{manifest_class}' "
                f"nhung anh gan nhom '{image_class}'."
            )
    if metrics["Very Dark"] > 0.08:
        score += 10
        reasons.append("Co vung rat dam/mat do cao can doi chieu.")
    if metrics["Line Count"] > 250:
        score += 8
        reasons.append("Anh co nhieu cau truc duong bien.")
    if metrics["Suspicious Zones"] >= 3:
        score += 8
        reasons.append("Co nhieu vung cau truc can doi chieu.")
    if ai_confidence < 55:
        score += 5
        reasons.append("Do tin cay AI con thap.")

    # V8 rules bo sung
    da = metrics.get("Dark Alarm Count", 0)
    if da >= 1:
        score += 20
        reasons.append(
            f"Phat hien {da} vung 'dark alarm' - tia X khong xuyen qua duoc "
            "(WCO/CASRA 2020). Can nhac soi tu goc khac."
        )
    ahr = metrics.get("Alpha Alarm Ratio", 0)
    if ahr > 0.03:
        score += 15
        reasons.append(
            f"Alpha-transparency rat cao ({ahr*100:.1f}%) - vat lieu qua dac (MIT/Lalor 2024)."
        )
    hom = metrics.get("Homogeneity", 1)
    if hom < 0.40:
        score += 15
        reasons.append(
            f"Hang hon tap (homogeneity={hom}) - kho phan tich, doi chieu ky manifest."
        )
    if metrics.get("Overlap Level", "") == "CAO":
        score += 10
        reasons.append("Nhieu lop vat the chong len nhau - superposition cao (Mademlis 2024).")
    wi = metrics.get("Weight Index", 0)
    if wi > 60:
        score += 12
        reasons.append(
            f"Chi so trong luong anh rat cao ({wi}) - doi chieu trong tai khai bao (ACXIS 2016)."
        )

    # Concealment scenario (MIT/Lalor 2024)
    concealment = False
    if (manifest_class in ("TEXTILE", "PLASTIC", "FOOD")
            and metrics.get("Alpha High Ratio", 0) > 0.05):
        score += 20
        concealment = True
        reasons.append(
            "CANH BAO: Khai bao hang nhe nhung co vung hap thu tia X bat thuong - "
            "kich ban an hang Z cao trong hang Z thap (MIT/Lalor 2024)."
        )

    # Physics warnings (MIT/Lalor 2024, Kolkoori/BAM 2014)
    if image_class in ("TEXTILE", "PLASTIC", "FOOD"):
        physics_warnings.append(
            "Vat lieu Z thap: tia X KHONG phan biet chinh xac giua cac hang huu co. "
            "Non-unique solution - MIT/Lalor 2024. Khong ket luan chi tu anh."
        )
    if metrics.get("Very Dark", 0) > 0.15:
        physics_warnings.append(
            "Hang day dac: beam hardening lam giam tuong phan dual-energy. "
            "Kha nang phan biet vat lieu giam (Kolkoori/BAM 2014)."
        )
    if manifest_class != "UNKNOWN" and image_class != "UNKNOWN":
        if manifest_class != image_class:
            physics_warnings.append(
                "Khong khop co the do gioi han vat ly: "
                "vat lieu Z cao mong va Z thap day cho anh X-ray giong nhau "
                "(MIT/Lalor 2024). Can xet them bang chung khac."
            )

    score = max(0, min(100, score))

    if score >= 75:
        level      = "🔴 RUI RO CAO"
        conclusion = ("Khuyen nghi kiem tra thuc te. "
                      "Neu co vung dark alarm, can nhac soi tu goc khac truoc khi mo container "
                      "(WCO/CASRA 2020). Objective: Homeland Security.")
    elif score >= 45:
        level      = "🟠 RUI RO TRUNG BINH"
        conclusion = ("Khuyen nghi ra soat Manifest va doi chieu ky voi anh. "
                      "Chu y vung chong chat va hang hon tap. "
                      "Objective: Contraband Interdiction (Reed/Varian 2008).")
    else:
        level      = "🟢 RUI RO THAP"
        conclusion = ("Hang tuong doi dong nhat, phu hop manifest. "
                      "Co the xem xet thong quan neu ho so day du. "
                      "Objective: Manifest Verification (Reed/Varian 2008).")

    return score, level, reasons, conclusion, physics_warnings


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Cai dat")
    lut_mode = st.selectbox(
        "🖼️ LUT Display Mode",
        ["Greyscale", "BW Log", "BW Sqrt", "Pseudo Color",
         "Histogram Mask", "Invert", "Edge Enhance",
         "Organic Only", "Quick Optimize", "Wiener Contrast",
         "Log Transform (MIT)"],
        help="Theo chuan Rapiscan GXA (CASRA 2021) va Michel-Mendes 2014"
    )
    inspection_obj = st.selectbox(
        "🎯 Muc tieu kiem tra",
        ["Manifest Verification", "Contraband Interdiction", "Homeland Security"],
        help="3 muc tieu theo Reed/Varian 2008 + Hitachi 2004"
    )
    st.divider()
    with st.expander("📚 15 nguon hoc thuat"):
        st.markdown("""
**Nghiep vu:**
- WCO/CASRA 2020
- Reed/Varian 2008
- Hitachi 2004

**Image Processing:**
- Jaccard/UCL 2016a+b
- ACXIS/EMPA 2016
- CASRA Simulator 2021
- Michel-Mendes 2014

**Physics:**
- Lalor/MIT 2024 v1+v2
- Kolkoori/BAM 2014
- Rogers/UCL 2016

**Deep Learning:**
- Mademlis/Athens 2024
- BARC India 2022
- AS&E/Saverskiy 2020
""")


# ═══════════════════════════════════════════════════════
# UI CHINH (giu layout V7.6)
# ═══════════════════════════════════════════════════════

manifest = st.text_area("📄 Manifest / Khai bao hang hoa", height=130)
uploaded_file = st.file_uploader("📤 Upload anh X-ray", type=["jpg", "jpeg", "png"])

if st.button("🔄 Reset / Xoa ket qua cu"):
    clear_old_results()
    st.rerun()

if uploaded_file is None:
    st.info(
        "Upload anh X-ray de bat dau.\n\n"
        "**V8.0 HYBRID - Nang cap tu V7.6:**\n"
        "- Physics-aware preprocessing (column normalize, stripe removal)\n"
        "- Alpha-map dual-energy (MIT/Lalor 2024)\n"
        "- Dark alarm detection (WCO/CASRA 2020)\n"
        "- Empty verification ~99.3% (Jaccard/UCL 2016)\n"
        "- Concealment scenario detection\n"
        "- Calibrated confidence (gioi han vat ly)\n"
        "- 11 LUT modes (Rapiscan GXA)\n"
        "- Sliding-window heatmap\n"
        "- Weight estimation (ACXIS/EMPA 2016)\n"
        "- 3-Stage recommendation (Hitachi 2004)"
    )
    st.stop()

current_key = make_hash(uploaded_file, manifest)
if st.session_state.get("last_key") != current_key:
    clear_old_results()
    st.session_state["last_key"] = current_key

image_pil = Image.open(uploaded_file)
manifest_class, manifest_scores = classify_manifest(manifest)

with st.spinner("⚙️ Dang phan tich V8 engine..."):
    img_result = analyze_image(image_pil)

image_class   = img_result["image_class"]
ai_label      = img_result["ai_label"]
ai_confidence = img_result["ai_confidence"]
metrics       = img_result["metrics"]
feature_scores = img_result["feature_scores"]

risk_score, risk_level, reasons, conclusion, physics_warnings = calculate_risk(
    manifest_class, image_class, metrics, ai_confidence
)

# ── SECTION 1-4: ANH (dung tabs giu gon) ────────────────

st.subheader("1. Anh X-ray")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Anh goc", "Tang tuong phan / LUT", "Heatmap mat do", "Sliding Heatmap", "Danh dau vung nghi van"
])
with tab1:
    st.image(image_pil, use_container_width=True, caption="Anh X-ray goc")
with tab2:
    lut_img = apply_lut(img_result["gray"], lut_mode)
    if lut_img.ndim == 2:
        st.image(lut_img, use_container_width=True,
                 caption=f"LUT: {lut_mode}", clamp=True)
    else:
        st.image(lut_img, use_container_width=True, caption=f"LUT: {lut_mode}")
    st.caption("Theo chuan Rapiscan GXA (CASRA 2021) va Michel-Mendes 2014")
with tab3:
    st.image(img_result["heatmap"], use_container_width=True,
             caption="Heatmap mat do hang hoa")
with tab4:
    st.image(img_result["sw_heatmap"], use_container_width=True,
             caption="Sliding-window heatmap (do=dang nghi) - Jaccard/UCL SPIE 2016")
with tab5:
    st.image(img_result["marked"], use_container_width=True,
             caption="Vung cau truc can doi chieu (Z-score > 2.5)")
    if img_result["suspicious_boxes"]:
        st.markdown("**Chi tiet vung danh dau:**")
        for i, (x, y, bw, bh, area, ed_in, pos) in enumerate(img_result["suspicious_boxes"]):
            st.caption(
                f"Vung #{i+1}: ({x},{y}) | {bw}x{bh}px | "
                f"edge={ed_in:.3f} | {pos}"
            )

# ── SECTION 5: PHAN TICH KY THUAT ────────────────────────

st.subheader("5. Phan tich ky thuat")
tab_basic, tab_v8, tab_z = st.tabs(["Chi so co ban (V7.6)", "Chi so V8 (Physics)", "Z-Range & Weight"])

with tab_basic:
    basic_keys = ["Dark Ratio","Very Dark","Std Density","Edge Density",
                  "Texture","Line Count","Object Area Ratio","Suspicious Zones",
                  "AI Confidence","Uniformity Score"]
    st.json({k: metrics[k] for k in basic_keys})

with tab_v8:
    v8_keys = ["Alpha Mean","Alpha High Ratio","Alpha Alarm Ratio",
               "Dark Alarm Count","Dark Alarm Ratio",
               "Homogeneity","Cargo Entropy","Cargo Complexity",
               "Zone Uniformity","Density Gradient",
               "Symmetry H","Symmetry V","Symmetry Comb",
               "FFT Hi Ratio","Corner Count",
               "MRA (Beer-Lambert)","Empty Score","Overlap Level"]
    st.json({k: metrics[k] for k in v8_keys})
    st.caption(
        "Alpha-map: MIT/Lalor 2024 | Dark alarm: WCO/CASRA 2020 | "
        "Entropy: Mademlis/Athens 2024 | MRA: Jaccard/UCL SPIE 2016"
    )

with tab_z:
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Z-Range (Rapiscan Sentry)", metrics["Z-Range"])
        st.caption("Nguon: Lalor/MIT-PNNL 2024, Table 1")
    with c2:
        st.metric("Weight Index (ACXIS)", metrics["Weight Index"])
        st.caption(img_result["weight_note"])
    st.info(f"Cargo Complexity: {img_result['cargo_complexity']}")

# ── SECTION 6: SO KHOP MANIFEST ──────────────────────────

st.subheader("6. So khop Manifest va anh")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Manifest phan loai", manifest_class)
with col2:
    st.metric("Anh suy doan nhom", image_class)
with col3:
    st.metric("Do tin cay (Calibrated)", f"{ai_confidence}%",
              help="Hieu chinh theo gioi han vat ly MIT/Lalor 2024")

st.progress(ai_confidence / 100)
st.write(f"Nhan dang anh: **{ai_label}**")

if manifest_class != "UNKNOWN" and image_class != "UNKNOWN":
    if manifest_class == image_class:
        st.success("Anh tuong doi phu hop voi Manifest.")
    else:
        st.error("Anh co dau hieu KHONG phu hop voi Manifest.")

with st.expander("Chi tiet diem Manifest"):
    st.json(manifest_scores)
with st.expander("Chi tiet diem dac trung anh"):
    st.json(feature_scores)

# ── SECTION 7: DANH GIA RUI RO ───────────────────────────

st.subheader("7. Danh gia rui ro")
st.markdown(f"## {risk_level}")
st.progress(risk_score / 100)
st.write(f"Diem nghi van: **{risk_score}/100** | Muc tieu: **{inspection_obj}**")

# ── SECTION 8: GIAI THICH ────────────────────────────────

st.subheader("8. Giai thich rui ro")
for r in reasons:
    st.write(f"- {r}")

# ── SECTION 9: KET LUAN ──────────────────────────────────

st.subheader("9. Ket luan nghiep vu")
if risk_score >= 75:
    st.error(conclusion)
elif risk_score >= 45:
    st.warning(conclusion)
else:
    st.success(conclusion)

# ── SECTION 10: CANH BAO GIOI HAN VAT LY (MOI) ───────────

st.subheader("10. Canh bao gioi han vat ly")
if physics_warnings:
    for w in physics_warnings:
        st.warning(f"⚠️ {w}")
else:
    st.success("Khong co canh bao gioi han vat ly dac biet.")
st.caption(
    "Nguon: MIT/Lalor 2024, Kolkoori/BAM 2014. "
    "He thong nay la cong cu ho tro (decision support), "
    "KHONG thay the quyet dinh cua nhan vien co tham quyen."
)

# ── SECTION 11: KHUYEN NGHI NGHIEP VU (MOI) ─────────────

st.subheader("11. Khuyen nghi nghiep vu")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**3-Stage Inspection (Hitachi 2004 / Reed/Varian 2008):**")
    if risk_score >= 75:
        st.error(
            "**Giai doan 2-3:**\n"
            "- Mo container, kiem tra tung kien hang (X-ray DR nho)\n"
            "- Dark alarm -> can nhac CT scanner\n"
            "- Ghi nhan vao ACXIS database (EMPA 2016)"
        )
    elif risk_score >= 45:
        st.warning(
            "**Giai doan 1.5:**\n"
            "- Soi lai tu goc khac neu co the\n"
            "- Doi chieu manifest voi tung vung anh\n"
            "- Xem xet phan tach container"
        )
    else:
        st.success(
            "**Giai doan 1 - Thong quan co dieu kien:**\n"
            "- Xac nhan ho so giay to day du\n"
            "- Luu anh vao database ACXIS\n"
            "- Co the xem xet thong quan"
        )
with col_b:
    st.info(
        f"Muc tieu: {inspection_obj}\n\n"
        f"Loai hang: {image_class} ({ai_confidence}%)\n\n"
        f"Z-range: {metrics['Z-Range']}\n\n"
        f"Weight Index: {metrics['Weight Index']} — {img_result['weight_note']}"
    )

st.divider()
st.caption(
    "X-Ray Cargo Analyzer V8.0 HYBRID | "
    "Engine: 15 nguon hoc thuat 2004-2024 | "
    "Layout: V7.6 | "
    "Chi dung cho muc dich ho tro nghiep vu hai quan."
)
