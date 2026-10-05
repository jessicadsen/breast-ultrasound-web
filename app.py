import time

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Breast Ultrasound Segmentation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #0D0D0D;
            color: #F2F2F2;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 3.2rem;
            padding-bottom: 2.5rem;
        }

        .brand {
            font-size: 0.68rem;
            letter-spacing: 3px;
            color: #777777;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .title {
            font-size: 2.75rem;
            line-height: 1.05;
            font-weight: 500;
            letter-spacing: -1.5px;
            color: #F4F4F4;
            margin-bottom: 0.65rem;
        }

        .subtitle {
            color: #858585;
            font-size: 0.92rem;
            line-height: 1.55;
            margin-bottom: 2.5rem;
            max-width: 760px;
        }

        .section-label {
            font-size: 0.66rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #777777;
            margin-bottom: 0.75rem;
        }

        .input-panel {
            background: #121212;
            border: 1px solid #292929;
            border-radius: 7px;
            padding: 1.25rem 1.35rem 1.1rem;
            margin-bottom: 1.2rem;
        }

        [data-testid="stFileUploader"] {
            background: #151515;
            border: 1px solid #2C2C2C;
            border-radius: 6px;
            padding: 0.85rem;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #505050;
        }

        .threshold-box {
            background: #151515;
            border: 1px solid #292929;
            border-radius: 6px;
            padding: 0.8rem 1.1rem;
            margin-top: 1rem;
        }

        .stButton {
            display: flex;
            justify-content: center;
            margin-top: 1.15rem;
            margin-bottom: 2rem;
        }

        .stButton > button {
            width: 230px;
            height: 42px;
            background: #E8E8E8;
            color: #111111;
            border: none;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 1.5px;
            transition: 0.2s ease;
        }

        .stButton > button:hover {
            background: #FFFFFF;
            color: #000000;
        }

        .analysis-container {
            background: #131313;
            border: 1px solid #292929;
            border-radius: 7px;
            padding: 1.7rem;
            margin-top: 0.5rem;
        }

        .analysis-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #292929;
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
        }

        .analysis-title {
            font-size: 0.72rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #AAAAAA;
        }

        .analysis-status {
            font-size: 0.65rem;
            color: #666666;
            letter-spacing: 1px;
        }

        .image-label {
            font-size: 0.65rem;
            letter-spacing: 1.8px;
            text-transform: uppercase;
            color: #777777;
            margin-bottom: 0.65rem;
        }

        .results {
            border-top: 1px solid #292929;
            margin-top: 1.8rem;
            padding-top: 1.5rem;
        }

        .metric-card {
            background: #171717;
            border: 1px solid #292929;
            border-radius: 5px;
            padding: 0.85rem 0.95rem;
            min-height: 76px;
        }

        .result-label {
            font-size: 0.59rem;
            letter-spacing: 1.4px;
            text-transform: uppercase;
            color: #666666;
            margin-bottom: 0.35rem;
        }

        .result-value {
            font-size: 1.05rem;
            font-weight: 500;
            color: #EEEEEE;
        }

        .result-subtext {
            color: #707070;
            font-size: 0.68rem;
            margin-top: 0.25rem;
        }

        .warning-box {
            background: #191919;
            border: 1px solid #383838;
            border-radius: 5px;
            padding: 0.9rem 1.05rem;
            margin-top: 1rem;
            color: #AAAAAA;
            font-size: 0.78rem;
            line-height: 1.5;
        }

        .disclaimer {
            color: #5F5F5F;
            font-size: 0.66rem;
            line-height: 1.5;
            margin-top: 1.2rem;
        }

        .footer {
            text-align: center;
            color: #444444;
            font-size: 0.62rem;
            letter-spacing: 0.8px;
            margin-top: 2.5rem;
            padding-top: 1.25rem;
            border-top: 1px solid #1F1F1F;
        }

        #MainMenu,
        footer {
            visibility: hidden;
        }

        header {
            background: transparent !important;
        }

        [data-testid="stMetricValue"] {
            color: #EEEEEE;
        }

        [data-testid="stCaptionContainer"] {
            color: #707070;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()
except Exception as exc:
    st.error("The YOLO26 model could not be loaded.")
    st.caption(f"Model loading error: {exc}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="brand">YOLO26 / ACADEMIC RESEARCH</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title">Deep Learning-Based Segmentation and Classification<br>of Breast Lesions in Ultrasound Datasets</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'YOLO26-based instance segmentation for breast ultrasound images. '
    'Upload an image to obtain the predicted lesion class, confidence, '
    'segmentation coverage, and inference information.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# IMAGE INPUT
# ============================================================

st.markdown('<div class="section-label">01 / IMAGE INPUT</div>', unsafe_allow_html=True)

st.markdown('<div class="input-panel">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload ultrasound image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# CONFIDENCE THRESHOLD
# ============================================================

st.markdown(
    '<div class="section-label" style="margin-top:1.1rem;">'
    'CONFIDENCE THRESHOLD'
    '</div>',
    unsafe_allow_html=True,
)

confidence_threshold = st.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.25,
    step=0.05,
    label_visibility="collapsed",
)

st.markdown(
    f'<div class="threshold-box">'
    f'<span style="color:#888888;font-size:0.76rem;">'
    f'Predictions displayed at confidence ≥ '
    f'<strong style="color:#E5E5E5;">{confidence_threshold:.2f}</strong>'
    f'</span></div>',
    unsafe_allow_html=True,
)


# ============================================================
# ANALYZE
# ============================================================

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    if st.button("ANALYZE IMAGE"):
        start_time = time.perf_counter()

        with st.spinner("Processing ultrasound image..."):
            results = model(image, conf=confidence_threshold, verbose=False)

        inference_time = time.perf_counter() - start_time
        result = results[0]
        result_image = result.plot()

        # ====================================================
        # ANALYSIS CONTAINER
        # ====================================================

        st.markdown('<div class="analysis-container">', unsafe_allow_html=True)

        st.markdown(
            '<div class="analysis-header">'
            '<div class="analysis-title">02 / IMAGE COMPARISON</div>'
            '<div class="analysis-status">ANALYSIS COMPLETE</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ====================================================
        # IMAGE COMPARISON
        # ====================================================

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="image-label">Original Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown('<div class="image-label">Segmentation Result</div>', unsafe_allow_html=True)
            st.image(result_image, use_container_width=True)

        # ====================================================
        # MODEL OUTPUT
        # ====================================================

        st.markdown('<div class="results">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">03 / MODEL OUTPUT</div>', unsafe_allow_html=True)

        image_width, image_height = image.size

        if result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.detach().cpu().tolist()
            class_ids = result.boxes.cls.detach().cpu().tolist()
            instances = len(result.boxes)

            best_index = confidences.index(max(confidences))
            confidence = float(confidences[best_index])
            class_id = int(class_ids[best_index])
            class_name = model.names[class_id]

            # Class distribution across all detected instances.
            class_counts = {}
            for detected_id in class_ids:
                detected_name = model.names[int(detected_id)]
                class_counts[detected_name] = class_counts.get(detected_name, 0) + 1

            # Bounding-box dimensions for the highest-confidence detection.
            best_box = result.boxes.xyxy[best_index].detach().cpu().tolist()
            x1, y1, x2, y2 = best_box
            bbox_width = max(0.0, x2 - x1)
            bbox_height = max(0.0, y2 - y1)
            bbox_area = bbox_width * bbox_height
            bbox_coverage = (bbox_area / (image_width * image_height)) * 100

            # Calculate a union mask after resizing to the original image size.
            # This avoids comparing mask pixels from the model's internal mask
            # resolution directly with the uploaded image dimensions.
            mask_coverage = None
            mask_pixels = None

            if result.masks is not None and len(result.masks.data) > 0:
                import torch.nn.functional as F

                masks = result.masks.data.float().unsqueeze(1)
                resized_masks = F.interpolate(
                    masks,
                    size=(image_height, image_width),
                    mode="nearest",
                ).squeeze(1)

                union_mask = (resized_masks > 0.5).any(dim=0)
                mask_pixels = int(union_mask.sum().item())
                mask_coverage = (mask_pixels / (image_width * image_height)) * 100

            # Main result cards.
            r1, r2, r3, r4 = st.columns(4)

            cards = [
                ("Detected Class", class_name, "Highest-confidence prediction"),
                ("Confidence", f"{confidence:.1%}", "Highest-confidence detection"),
                ("Instances", str(instances), "Detected lesion instances"),
                ("Inference Time", f"{inference_time * 1000:.0f} ms", "Model inference"),
            ]

            for column, (label, value, subtext) in zip((r1, r2, r3, r4), cards):
                with column:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="result-label">{label}</div>'
                        f'<div class="result-value">{value}</div>'
                        f'<div class="result-subtext">{subtext}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Segmentation details.
            st.markdown(
                '<div style="margin-top:1.5rem;">'
                '<div class="section-label">SEGMENTATION DETAILS</div>',
                unsafe_allow_html=True,
            )

            s1, s2, s3, s4 = st.columns(4)

            detail_cards = [
                ("Mask Detected", "Yes" if result.masks is not None else "No", "Instance mask output"),
                (
                    "Mask Coverage",
                    f"{mask_coverage:.2f}%" if mask_coverage is not None else "N/A",
                    "Union of predicted masks",
                ),
                ("Bounding Box", f"{bbox_width:.0f} × {bbox_height:.0f} px", "Top prediction"),
                ("Image Resolution", f"{image_width} × {image_height}", "Uploaded image"),
            ]

            for column, (label, value, subtext) in zip((s1, s2, s3, s4), detail_cards):
                with column:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="result-label">{label}</div>'
                        f'<div class="result-value">{value}</div>'
                        f'<div class="result-subtext">{subtext}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Detected-class summary, useful during the project defense.
            if class_counts:
                summary = " · ".join(
                    f"{name}: {count}" for name, count in class_counts.items()
                )
                st.markdown(
                    f'<div class="warning-box">'
                    f'<strong style="color:#D5D5D5;">Detected classes:</strong> {summary}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        else:
            st.markdown(
                '<div class="result-label">DETECTION STATUS</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="result-value">No segmentation detected</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="warning-box">'
                f'The model did not produce a prediction above the current '
                f'confidence threshold of {confidence_threshold:.2f}. '
                f'Try a lower threshold if appropriate, but interpret low-confidence '
                f'predictions cautiously.'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div style="margin-top:1.5rem;">'
                '<div class="section-label">IMAGE / INFERENCE INFORMATION</div>',
                unsafe_allow_html=True,
            )

            n1, n2, n3 = st.columns(3)

            no_detection_cards = [
                ("Image Resolution", f"{image_width} × {image_height}"),
                ("Threshold", f"{confidence_threshold:.2f}"),
                ("Inference Time", f"{inference_time * 1000:.0f} ms"),
            ]

            for column, (label, value) in zip((n1, n2, n3), no_detection_cards):
                with column:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="result-label">{label}</div>'
                        f'<div class="result-value">{value}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown(
            '<div class="disclaimer">'
            'Research prototype for academic use. Predictions are model outputs and '
            'should not be interpreted as a clinical diagnosis.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'BREAST ULTRASOUND SEGMENTATION &nbsp;·&nbsp; YOLO26'
    '</div>',
    unsafe_allow_html=True,
)
