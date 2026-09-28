import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Breast Ultrasound Segmentation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .stApp {
        background: #0D0D0D;
        color: #F2F2F2;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 4rem;
        padding-bottom: 3rem;
    }

    /* HEADER */

    .brand {
        font-size: 0.72rem;
        letter-spacing: 3px;
        color: #777777;
        text-transform: uppercase;
        margin-bottom: 1.4rem;
    }

    .title {
        font-size: 3rem;
        line-height: 1.05;
        font-weight: 500;
        letter-spacing: -1.5px;
        color: #F4F4F4;
        margin-bottom: 0.7rem;
    }

    .subtitle {
        color: #777777;
        font-size: 0.95rem;
        margin-bottom: 3rem;
    }

    /* SECTION LABEL */

    .section-label {
        font-size: 0.68rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #777777;
        margin-bottom: 0.8rem;
    }

    /* UPLOAD */

    [data-testid="stFileUploader"] {
        background: #151515;
        border: 1px solid #2C2C2C;
        border-radius: 6px;
        padding: 1.1rem;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #505050;
    }

    /* THRESHOLD */

    .threshold-box {
        background: #131313;
        border: 1px solid #292929;
        border-radius: 6px;
        padding: 1rem 1.3rem;
        margin-top: 1.2rem;
    }

    /* BUTTON */

    .stButton {
        display: flex;
        justify-content: center;
        margin-top: 1.4rem;
        margin-bottom: 3rem;
    }

    .stButton > button {
        width: 220px;
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

    /* ANALYSIS */

    .analysis-container {
        background: #131313;
        border: 1px solid #292929;
        border-radius: 6px;
        padding: 2rem;
        margin-top: 0.5rem;
    }

    .analysis-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #292929;
        padding-bottom: 1rem;
        margin-bottom: 1.7rem;
    }

    .analysis-title {
        font-size: 0.72rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #AAAAAA;
    }

    .analysis-status {
        font-size: 0.68rem;
        color: #666666;
        letter-spacing: 1px;
    }

    /* IMAGES */

    .image-label {
        font-size: 0.67rem;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: #777777;
        margin-bottom: 0.7rem;
    }

    /* RESULTS */

    .results {
        border-top: 1px solid #292929;
        margin-top: 2rem;
        padding-top: 1.7rem;
    }

    .result-label {
        font-size: 0.62rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #666666;
        margin-bottom: 0.4rem;
    }

    .result-value {
        font-size: 1.15rem;
        font-weight: 500;
        color: #EEEEEE;
    }

    .warning-box {
        background: #191919;
        border: 1px solid #383838;
        border-radius: 5px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        color: #AAAAAA;
        font-size: 0.82rem;
        line-height: 1.5;
    }

    /* FOOTER */

    .footer {
        text-align: center;
        color: #444444;
        font-size: 0.65rem;
        letter-spacing: 0.8px;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #1F1F1F;
    }

    /* HIDE STREAMLIT UI */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="brand">US-SEG / ACADEMIC RESEARCH</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="title">Breast Ultrasound<br>Segmentation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'YOLO26-based instance segmentation for breast ultrasound images'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE INPUT
# ============================================================

st.markdown(
    '<div class="section-label">01 / IMAGE INPUT</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload ultrasound image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# CONFIDENCE THRESHOLD
# ============================================================

st.markdown(
    '<div class="section-label" style="margin-top:1.5rem;">'
    'CONFIDENCE THRESHOLD'
    '</div>',
    unsafe_allow_html=True
)

confidence_threshold = st.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.25,
    step=0.05,
    label_visibility="collapsed"
)

st.caption(
    f"Only predictions with confidence ≥ {confidence_threshold:.2f} "
    "will be displayed."
)


# ============================================================
# ANALYZE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    if st.button("ANALYZE IMAGE"):

        start_time = time.perf_counter()

        with st.spinner("Processing ultrasound image..."):

            results = model(
                image,
                conf=confidence_threshold
            )

        inference_time = time.perf_counter() - start_time

        result = results[0]
        result_image = result.plot()


        # ====================================================
        # ANALYSIS CONTAINER
        # ====================================================

        st.markdown(
            '<div class="analysis-container">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="analysis-header">'
            '<div class="analysis-title">02 / IMAGE COMPARISON</div>'
            '<div class="analysis-status">ANALYSIS COMPLETE</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # IMAGE COMPARISON
        # ====================================================

        col1, col2 = st.columns(2, gap="large")

        with col1:

            st.markdown(
                '<div class="image-label">Original Image</div>',
                unsafe_allow_html=True
            )

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown(
                '<div class="image-label">Segmentation Result</div>',
                unsafe_allow_html=True
            )

            st.image(
                result_image,
                use_container_width=True
            )


        # ====================================================
        # MODEL OUTPUT
        # ====================================================

        st.markdown(
            '<div class="results">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-label">03 / MODEL OUTPUT</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # DETECTION EXISTS
        # ====================================================

        if result.boxes is not None and len(result.boxes) > 0:

            confidences = result.boxes.conf.tolist()
            class_ids = result.boxes.cls.tolist()

            best_index = confidences.index(max(confidences))

            confidence = confidences[best_index]
            class_id = int(class_ids[best_index])

            class_name = model.names[class_id]

            instances = len(result.boxes)


            # ------------------------------------------------
            # BASIC RESULTS
            # ------------------------------------------------

            r1, r2, r3, r4 = st.columns(4)

            with r1:

                st.markdown(
                    '<div class="result-label">Detected Class</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="result-value">{class_name}</div>',
                    unsafe_allow_html=True
                )

            with r2:

                st.markdown(
                    '<div class="result-label">Confidence</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="result-value">{confidence:.1%}</div>',
                    unsafe_allow_html=True
                )

            with r3:

                st.markdown(
                    '<div class="result-label">Instances</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="result-value">{instances}</div>',
                    unsafe_allow_html=True
                )

            with r4:

                st.markdown(
                    '<div class="result-label">Inference Time</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="result-value">'
                    f'{inference_time * 1000:.0f} ms'
                    f'</div>',
                    unsafe_allow_html=True
                )


            # ------------------------------------------------
            # MASK INFORMATION
            # ------------------------------------------------

            if result.masks is not None:

                # Image dimensions
                image_width, image_height = image.size

                total_pixels = image_width * image_height

                # Get segmentation masks
                masks = result.masks.data

                # Calculate mask coverage
                mask_pixels = 0

                for mask in masks:

                    mask_resized = (
                        mask.float()
                        .cpu()
                        .numpy()
                    )

                    mask_pixels += (mask_resized > 0.5).sum()

                mask_coverage = (
                    mask_pixels / total_pixels
                ) * 100


                st.markdown(
                    '<div style="margin-top:2rem;">'
                    '<div class="section-label">'
                    'SEGMENTATION DETAILS'
                    '</div>',
                    unsafe_allow_html=True
                )

                s1, s2, s3 = st.columns(3)

                with s1:

                    st.markdown(
                        '<div class="result-label">'
                        'Mask Detected'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '<div class="result-value">Yes</div>',
                        unsafe_allow_html=True
                    )

                with s2:

                    st.markdown(
                        '<div class="result-label">'
                        'Mask Coverage'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="result-value">'
                        f'{mask_coverage:.2f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with s3:

                    st.markdown(
                        '<div class="result-label">'
                        'Image Resolution'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="result-value">'
                        f'{image_width} × {image_height}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.markdown('</div>', unsafe_allow_html=True)


        # ====================================================
        # NO DETECTION
        # ====================================================

        else:

            st.markdown(
                '<div class="result-label">'
                'DETECTION STATUS'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="result-value">'
                'No segmentation detected'
                '</div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # IMPORTANT INFORMATION
            # ------------------------------------------------

            st.markdown(
                f'''
                <div class="warning-box">

                    The model did not produce a prediction
                    above the current confidence threshold of {confidence_threshold:.2f}.

                </div>
                ''',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # IMAGE INFORMATION
            # ------------------------------------------------

            image_width, image_height = image.size

            st.markdown(
                '<div style="margin-top:1.5rem;">'
                '<div class="section-label">'
                'IMAGE / INFERENCE INFORMATION'
                '</div>',
                unsafe_allow_html=True
            )

            n1, n2, n3 = st.columns(3)

            with n1:

                st.markdown(
                    '<div class="result-label">Image Resolution</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="result-value">'
                    f'{image_width} × {image_height}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with n2:

                st.markdown(
                    '<div class="result-label">Threshold</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="result-value">'
                    f'{confidence_threshold:.2f}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with n3:

                st.markdown(
                    '<div class="result-label">Inference Time</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="result-value">'
                    f'{inference_time * 1000:.0f} ms'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)


        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '''
    <div class="footer">
        BREAST ULTRASOUND SEGMENTATION
        &nbsp;·&nbsp;
        PRAGMATISM
    </div>
    ''',
    unsafe_allow_html=True
)