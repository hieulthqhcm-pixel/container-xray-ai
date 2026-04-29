import streamlit as st
import cv2
import numpy as np
from PIL import Image
import hashlib

st.set_page_config(
    page_title="X-Ray Cargo Analyzer V7.6 MATCHING PROFESSIONAL",
    page_icon="📦",
    layout="wide"
)

st.title("📦 X-Ray Cargo Analyzer V7.6 MATCHING PROFESSIONAL")

st.caption(
    "Phân tích ảnh soi chiếu container bằng OpenCV local - ưu tiên so khớp Manifest và đặc trưng ảnh."
)

# =========================
# RESET
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

    for k in ["last_key"]:

        if k in st.session_state:
            del st.session_state[k]


# =========================
# MANIFEST CLASSIFY
# =========================

def classify_manifest(text):

    t = text.upper()

    groups = {

        "MACHINERY": [
            "EXCAVATOR",
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
            "FLOWERPOT",
            "POT",
            "TILE",
            "PORCELAIN",
            "STONE",
            "GRANITE",
            "MARBLE",
            "BRICK",
            "6913",
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
        ],

        "LIQUID_CHEMICAL": [
            "LIQUID",
            "CHEMICAL",
            "OIL",
            "PAINT",
            "INK",
            "ADHESIVE",
            "DRUM",
            "BARREL",
            "IBC",
            "TANK"
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
# IMAGE ANALYSIS
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

    roi_x1 = int(w * 0.12)
    roi_x2 = int(w * 0.88)

    roi_y1 = int(h * 0.12)
    roi_y2 = int(h * 0.88)

    roi_mask = np.zeros_like(gray)

    roi_mask[
        roi_y1:roi_y2,
        roi_x1:roi_x2
    ] = 255

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

    edges = cv2.Canny(
        blur,
        35,
        120
    )

    edges = cv2.bitwise_and(
        edges,
        roi_mask
    )

    dark_threshold = np.percentile(
        blur,
        28
    )

    very_dark_threshold = np.percentile(
        blur,
        12
    )

    dark_mask = np.where(
        blur < dark_threshold,
        255,
        0
    ).astype(np.uint8)

    very_dark_mask = np.where(
        blur < very_dark_threshold,
        255,
        0
    ).astype(np.uint8)

    dark_mask = cv2.bitwise_and(
        dark_mask,
        roi_mask
    )

    very_dark_mask = cv2.bitwise_and(
        very_dark_mask,
        roi_mask
    )

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

    contours, _ = cv2.findContours(
        object_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    suspicious_boxes = []

    object_area_sum = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < total_pixels * 0.0015:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw < 35 or bh < 35:
            continue

        if bh > bw * 3 and bw < w * 0.08:
            continue

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

        if edge_density_inside < 0.012:
            continue

        object_area_sum += area

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

    dark_ratio = np.sum(
        dark_mask > 0
    ) / total_pixels

    very_dark_ratio = np.sum(
        very_dark_mask > 0
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

    object_area_ratio = object_area_sum / total_pixels

    feature_scores = {

        "MACHINERY": 0,
        "CERAMIC": 0,
        "TEXTILE": 0,
        "PLASTIC": 0,
        "ELECTRONICS": 0,
        "LIQUID_CHEMICAL": 0
    }

    if line_count >= 180:
        feature_scores["MACHINERY"] += 35

    elif line_count >= 100:
        feature_scores["MACHINERY"] += 25

    if edge_density >= 0.020:
        feature_scores["MACHINERY"] += 25

    if texture >= 180:
        feature_scores["MACHINERY"] += 20

    if very_dark_ratio >= 0.06:
        feature_scores["MACHINERY"] += 10
        feature_scores["ELECTRONICS"] += 10

    if dark_ratio >= 0.18:
        feature_scores["CERAMIC"] += 25

    if edge_density < 0.018 and texture < 220:
        feature_scores["CERAMIC"] += 25

    if object_area_ratio >= 0.08 and line_count < 150:
        feature_scores["CERAMIC"] += 20

    if edge_density < 0.015:
        feature_scores["TEXTILE"] += 25

    if texture < 120:
        feature_scores["TEXTILE"] += 25

    if very_dark_ratio < 0.04:
        feature_scores["TEXTILE"] += 15

    if 0.08 <= dark_ratio <= 0.20:
        feature_scores["PLASTIC"] += 20

    if 100 <= texture <= 260:
        feature_scores["PLASTIC"] += 20

    if line_count < 180:
        feature_scores["PLASTIC"] += 10

    if texture >= 300:
        feature_scores["ELECTRONICS"] += 30

    if edge_density >= 0.028:
        feature_scores["ELECTRONICS"] += 25

    if line_count >= 220:
        feature_scores["ELECTRONICS"] += 15

    if edge_density < 0.014 and std_density < 45:
        feature_scores["LIQUID_CHEMICAL"] += 25

    if dark_ratio >= 0.12 and texture < 140:
        feature_scores["LIQUID_CHEMICAL"] += 20

    image_class = max(
        feature_scores,
        key=feature_scores.get
    )

    image_score = feature_scores[image_class]

    if image_score < 35:
        image_class = "UNKNOWN"

    ai_confidence = min(
        95,
        max(45, int(image_score))
    )

    label_map = {

        "MACHINERY":
            "MACHINERY / HEAVY STRUCTURE",

        "CERAMIC":
            "CERAMIC / STONE / DENSE GOODS",

        "TEXTILE":
            "TEXTILE / SOFT GOODS",

        "PLASTIC":
            "PLASTIC / LIGHT-MEDIUM GOODS",

        "ELECTRONICS":
            "ELECTRONICS / COMPLEX STRUCTURE",

        "LIQUID_CHEMICAL":
            "LIQUID / CHEMICAL / DRUM GOODS",

        "UNKNOWN":
            "UNKNOWN"
    }

    ai_label = label_map.get(
        image_class,
        "UNKNOWN"
    )

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
            f"DOI CHIEU {i+1}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

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

        "Object Area Ratio":
            round(float(object_area_ratio), 3),

        "Suspicious Zones":
            int(len(suspicious_boxes)),

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

        "feature_scores": feature_scores,

        "suspicious_boxes": suspicious_boxes
    }