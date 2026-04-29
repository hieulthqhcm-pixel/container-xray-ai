

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

    # =========================
    # ROI GIỮA ẢNH
    # =========================

    roi_x1 = int(w * 0.18)
    roi_x2 = int(w * 0.82)

    roi_y1 = int(h * 0.18)
    roi_y2 = int(h * 0.82)

    roi_mask = np.zeros_like(gray)

    roi_mask[
        roi_y1:roi_y2,
        roi_x1:roi_x2
    ] = 255

    # =========================
    # TĂNG TƯƠNG PHẢN
    # =========================

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

    # =========================
    # EDGE
    # =========================

    edges = cv2.Canny(
        blur,
        35,
        120
    )

    edges = cv2.bitwise_and(
        edges,
        roi_mask
    )

    # =========================
    # MASK VÙNG ĐẬM
    # =========================

    dark_threshold = np.percentile(
        blur,
        28
    )

    dark_mask = np.where(
        blur < dark_threshold,
        255,
        0
    ).astype(np.uint8)

    dark_mask = cv2.bitwise_and(
        dark_mask,
        roi_mask
    )

    # =========================
    # OBJECT MASK
    # =========================

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

    # =========================
    # CONTOUR
    # =========================

    contours, _ = cv2.findContours(
        object_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    suspicious_boxes = []

    machinery_shapes = 0
    arm_shapes = 0
    cabin_shapes = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < total_pixels * 0.002:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # bỏ box nhỏ
        if bw < 45 or bh < 45:
            continue

        # bỏ chữ dọc
        if bh > bw * 3:
            continue

        # bỏ sát mép
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

        if edge_density_inside < 0.02:
            continue

        # =========================
        # PHÂN TÍCH SHAPE
        # =========================

        aspect_ratio = bw / (bh + 1)

        # dạng thân máy/cabin
        if 0.7 < aspect_ratio < 2.5:
            cabin_shapes += 1

        # dạng tay cần dài
        if aspect_ratio > 2.8:
            arm_shapes += 1

        machinery_shapes += 1

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

    # =========================
    # HEATMAP
    # =========================

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

    # =========================
    # DRAW
    # =========================

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
            f"MACHINERY AREA {i+1}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

    # =========================
    # METRICS
    # =========================

    dark_ratio = np.sum(
        dark_mask > 0
    ) / total_pixels

    very_dark_ratio = np.sum(
        blur < np.percentile(blur, 12)
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

    # =========================
    # AI SUY ĐOÁN LOẠI HÀNG
    # =========================

    image_class = "UNKNOWN"

    ai_label = "UNKNOWN"

    ai_confidence = 50

    # =========================
    # NHẬN DIỆN MÁY XÚC
    # =========================

    if (
        line_count > 100
        and edge_density > 0.020
        and machinery_shapes >= 2
        and arm_shapes >= 1
    ):

        image_class = "MACHINERY"

        ai_label = "USED EXCAVATOR"

        ai_confidence = 90

    # =========================
    # XE NÂNG / XE CÔNG TRÌNH
    # =========================

    elif (
        line_count > 70
        and edge_density > 0.018
        and machinery_shapes >= 2
    ):

        image_class = "MACHINERY"

        ai_label = "HEAVY MACHINERY"

        ai_confidence = 80

    # =========================
    # GẠCH / CERAMIC
    # =========================

    elif (
        dark_ratio > 0.22
        and edge_density < 0.018
    ):

        image_class = "CERAMIC"

        ai_label = "CERAMIC / TILE"

        ai_confidence = 75

    # =========================
    # TEXTILE
    # =========================

    elif (
        texture < 80
        and dark_ratio < 0.14
    ):

        image_class = "TEXTILE"

        ai_label = "TEXTILE"

        ai_confidence = 70

    # =========================
    # PLASTIC
    # =========================

    elif (
        dark_ratio < 0.18
        and texture < 180
    ):

        image_class = "PLASTIC"

        ai_label = "PLASTIC GOODS"

        ai_confidence = 70

    # =========================
    # ELECTRONICS
    # =========================

    elif (
        edge_density > 0.035
        and texture > 300
    ):

        image_class = "ELECTRONICS"

        ai_label = "ELECTRONIC EQUIPMENT"

        ai_confidence = 75

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

        "Suspicious Zones":
            int(len(suspicious_boxes)),

        "Machinery Shapes":
            int(machinery_shapes),

        "Arm Shapes":
            int(arm_shapes),

        "Cabin Shapes":
            int(cabin_shapes),

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

        "suspicious_boxes": suspicious_boxes
    }