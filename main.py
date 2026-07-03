import os
import asyncio
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
)
from ultralytics import YOLO

from utils.feedback_logger import (
    save_feedback,
    get_feedback_stats,
)

# -------------------------------------------------------------------
# LOGGING FIXES
# -------------------------------------------------------------------

os.environ["AIOICE_LOG_LEVEL"] = "CRITICAL"

logging.getLogger("aioice").setLevel(logging.CRITICAL)
logging.getLogger("aioice.ice").setLevel(logging.CRITICAL)
logging.getLogger("aiortc").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# -------------------------------------------------------------------
# ASYNCIO ERROR FILTER
# -------------------------------------------------------------------


def _silence_known_teardown_errors(loop, context):

    exc = context.get("exception")

    message = context.get("message", "")

    if isinstance(exc, AttributeError):

        if "sendto" in str(exc):
            return

        if "call_exception_handler" in str(exc):
            return

    if "Task was destroyed but it is pending" in message:
        return

    loop.default_exception_handler(context)


def _install_exception_handler_on_current_loop():

    try:

        loop = asyncio.get_event_loop()

        loop.set_exception_handler(
            _silence_known_teardown_errors
        )

    except RuntimeError:
        pass


_install_exception_handler_on_current_loop()

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

MODEL_PATH = "model/best.pt"

CONFIDENCE_THRESHOLD = 0.5

GOOD_CLASS_NAME = "good"

SAMPLE_IMAGES_DIR = Path("sample_images")

COLOR_GOOD = (0, 200, 0)

COLOR_DEFECT = (0, 0, 255)

# -------------------------------------------------------------------
# STABLE RTC CONFIGURATION
# -------------------------------------------------------------------

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302",
                    "stun:stun1.l.google.com:19302",
                ]
            }
        ],
        "iceTransportPolicy": "all",
    }
)

# -------------------------------------------------------------------
# SAMPLE IMAGES
# -------------------------------------------------------------------

SAMPLE_IMAGES = [
    {"file": "sample1.jpg", "caption": "Delik"},
    {"file": "sample2.jpg", "caption": "Delik"},
    {"file": "sample3.jpg", "caption": "Jut"},
    {"file": "sample4.jpg", "caption": "Leke"},
]

# -------------------------------------------------------------------
# DEFECT CLASSES
# -------------------------------------------------------------------

DEFECT_CLASSES = [
    "good",
    "Delik",
    "Jut",
    "Leke",
    "pinhole",
    "blur",
    "misregistration",
    "ink_bleed",
    "ghosting",
    "uneven_deposit",
    "low_opacity",
]

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Garment Print Defect Detection",
    layout="wide",
)

# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------


@st.cache_resource
def load_model():

    if not Path(MODEL_PATH).exists():
        return None

    return YOLO(MODEL_PATH)

# -------------------------------------------------------------------
# IMAGE HELPERS
# -------------------------------------------------------------------


def pil_to_cv2(img: Image.Image):

    arr = np.array(img.convert("RGB"))

    return cv2.cvtColor(
        arr,
        cv2.COLOR_RGB2BGR,
    )

# -------------------------------------------------------------------
# RUN INFERENCE
# -------------------------------------------------------------------


def run_inference(model, image_bgr):

    results = model(
        image_bgr,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    annotated = results[0].plot()

    boxes = results[0].boxes

    class_names = results[0].names

    status = "GOOD"

    top_class = GOOD_CLASS_NAME

    top_conf = None

    if boxes is not None and len(boxes) > 0:

        confs = []

        names = []

        for box in boxes:

            cls_id = int(box.cls[0])

            conf = float(box.conf[0])

            name = class_names.get(
                cls_id,
                str(cls_id),
            )

            confs.append(conf)

            names.append(name)

        best_idx = int(np.argmax(confs))

        top_class = names[best_idx]

        top_conf = confs[best_idx]

        status = (
            "GOOD"
            if top_class.lower() == GOOD_CLASS_NAME
            else "DEFECTIVE"
        )

    return (
        status,
        top_class,
        top_conf,
        annotated,
    )

# -------------------------------------------------------------------
# DRAW STATUS BAR
# -------------------------------------------------------------------


def draw_status_bar(
    frame,
    status,
    top_class,
    conf,
    fps,
):

    h, w = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, 60),
        (30, 30, 30),
        -1,
    )

    frame = cv2.addWeighted(
        overlay,
        0.6,
        frame,
        0.4,
        0,
    )

    color = (
        COLOR_GOOD
        if status == "GOOD"
        else COLOR_DEFECT
    )

    text = f"STATUS: {status}"

    if status == "DEFECTIVE":

        text += f" ({top_class})"

    if conf is not None:

        text += f" {conf*100:.0f}%"

    cv2.putText(
        frame,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return frame

# -------------------------------------------------------------------
# VIDEO PROCESSOR
# -------------------------------------------------------------------


class DefectVideoProcessor:

    def __init__(self, model):

        self.model = model

        self.lock = threading.Lock()

        self.latest_frame_bgr = None

        self.latest_status = "GOOD"

        self.latest_class = GOOD_CLASS_NAME

        self.latest_conf = None

        self._prev_time = time.time()

        self._handler_installed = False

    def recv(self, frame):

        if not self._handler_installed:

            _install_exception_handler_on_current_loop()

            self._handler_installed = True

        img = frame.to_ndarray(format="bgr24")

        (
            status,
            top_class,
            top_conf,
            annotated,
        ) = run_inference(
            self.model,
            img,
        )

        now = time.time()

        fps = (
            1.0 / (now - self._prev_time)
            if now != self._prev_time
            else 0.0
        )

        self._prev_time = now

        annotated = draw_status_bar(
            annotated,
            status,
            top_class,
            top_conf,
            fps,
        )

        with self.lock:

            self.latest_frame_bgr = img.copy()

            self.latest_status = status

            self.latest_class = top_class

            self.latest_conf = top_conf

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24",
        )

# -------------------------------------------------------------------
# SAMPLE IMAGES SECTION
# -------------------------------------------------------------------


def sample_images_section():

    st.subheader("📷 Sample Images")

    cols = st.columns(4)

    for col, img in zip(cols, SAMPLE_IMAGES):

        img_path = SAMPLE_IMAGES_DIR / img["file"]

        with col:

            if img_path.exists():

                st.image(
                    str(img_path),
                    caption=img["caption"],
                    use_container_width=True,
                )

                with open(img_path, "rb") as f:

                    img_bytes = f.read()

                st.download_button(
                    label="⬇️ Download",
                    data=img_bytes,
                    file_name=img["file"],
                    mime="image/jpeg",
                    key=f"download_{img['file']}",
                    use_container_width=True,
                )

# -------------------------------------------------------------------
# FEEDBACK CONTROLS
# -------------------------------------------------------------------


def render_feedback_controls(
    image_bgr,
    top_class,
    top_conf,
    key_prefix,
):

    state_key = f"feedback_done_{key_prefix}"

    if state_key not in st.session_state:

        st.session_state[state_key] = False

    if st.session_state[state_key]:

        st.success("Feedback recorded.")

        return

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "👍 Correct",
            key=f"correct_{key_prefix}",
            use_container_width=True,
        ):

            save_feedback(
                image_bgr,
                top_class,
                top_conf,
                was_correct=True,
            )

            st.session_state[state_key] = True

            st.rerun()

    with col2:

        if st.button(
            "👎 Incorrect",
            key=f"incorrect_{key_prefix}",
            use_container_width=True,
        ):

            corrected = st.selectbox(
                "Correct label",
                DEFECT_CLASSES,
                key=f"select_{key_prefix}",
            )

            if st.button(
                "Submit",
                key=f"submit_{key_prefix}",
            ):

                save_feedback(
                    image_bgr,
                    top_class,
                    top_conf,
                    was_correct=False,
                    corrected_class=corrected,
                )

                st.session_state[state_key] = True

                st.rerun()

# -------------------------------------------------------------------
# UPLOAD MODE
# -------------------------------------------------------------------


def upload_mode(model):

    uploaded_file = st.file_uploader(
        "Upload shirt print image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:

        st.info("Upload an image.")

        return

    image = Image.open(uploaded_file)

    image_bgr = pil_to_cv2(image)

    with st.spinner("Running detection..."):

        (
            status,
            top_class,
            top_conf,
            annotated_bgr,
        ) = run_inference(
            model,
            image_bgr,
        )

    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original",
            use_container_width=True,
        )

    with col2:

        st.image(
            annotated_rgb,
            caption="Detection Result",
            use_container_width=True,
        )


        if status == "GOOD":

            st.success(
                "STATUS: GOOD"
                + (
                    f" ({top_conf*100:.1f}% confidence)"
                    if top_conf is not None
                    else ""
                )
            )

        else:

            st.error(
                f"STATUS: DEFECTIVE - {top_class}"
                + (
                    f" ({top_conf*100:.1f}% confidence)"
                    if top_conf is not None
                    else ""
                )
            )


    render_feedback_controls(
        image_bgr,
        top_class,
        top_conf,
        key_prefix="upload",
    )

# -------------------------------------------------------------------
# LIVE CAMERA MODE
# -------------------------------------------------------------------


def live_camera_mode(model):

    st.caption(
        "Click START and allow browser camera access."
    )

    ctx = webrtc_streamer(
        key="defect-detection-live",

        mode=WebRtcMode.SENDRECV,

        rtc_configuration=RTC_CONFIGURATION,

        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },

        video_processor_factory=lambda:
            DefectVideoProcessor(model),

        async_processing=False,
    )

    st.divider()

    if ctx.state.playing:

        st.success("✅ Camera connected")

    else:

        st.info("▶️ Click START above")

        return

    st.subheader("Capture current frame")

    if ctx.video_processor is None:

        st.warning("Waiting for frames...")

        return

    if st.button("📸 Capture current frame"):

        try:

            with ctx.video_processor.lock:

                frame = ctx.video_processor.latest_frame_bgr

                status = ctx.video_processor.latest_status

                top_class = ctx.video_processor.latest_class

                top_conf = ctx.video_processor.latest_conf

            if frame is None:

                st.warning("No frame available yet.")

                return

            st.session_state["captured_frame"] = frame

            st.session_state["captured_status"] = status

            st.session_state["captured_class"] = top_class

            st.session_state["captured_conf"] = top_conf

        except Exception as e:

            st.error(f"Capture failed: {e}")

            return

    if "captured_frame" in st.session_state:

        frame = st.session_state["captured_frame"]

        status = st.session_state["captured_status"]

        top_class = st.session_state["captured_class"]

        top_conf = st.session_state["captured_conf"]

        st.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            caption="Captured Frame",
            use_container_width=True,
        )


        if status == "GOOD":

            st.success(
                "STATUS: GOOD"
                + (
                    f" ({top_conf*100:.1f}% confidence)"
                    if top_conf is not None
                    else ""
                )
            )

        else:

            st.error(
                f"STATUS: DEFECTIVE - {top_class}"
                + (
                    f" ({top_conf*100:.1f}% confidence)"
                    if top_conf is not None
                    else ""
                )
            )



        render_feedback_controls(
            frame,
            top_class,
            top_conf,
            key_prefix="live",
        )

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------


def main():

    st.title("🧵 Garment Print Defect Detection")

    model = load_model()

    with st.sidebar:

        st.header("Model Status")

        if model is None:

            st.error(
                f"No model found at {MODEL_PATH}"
            )

            return

        st.success("Model loaded")

        st.header("Feedback Stats")

        stats = get_feedback_stats()

        st.metric("Total", stats["total"])

        col1, col2 = st.columns(2)

        col1.metric("Correct", stats["correct"])

        col2.metric("Incorrect", stats["incorrect"])

        st.header("Retraining")

        if st.button(
            "🔁 Run Retraining",
            use_container_width=True,
        ):

            with st.spinner("Running retraining..."):

                result = subprocess.run(
                    ["python", "retrain_from_feedback.py"],
                    capture_output=True,
                    text=True,
                )

            st.code(
                result.stdout + result.stderr
            )

    mode = st.radio(
        "Modes",
        [
            "Sample Images",
            "Upload Image",
            "Live Camera",
        ],
        horizontal=True,
    )

    st.divider()

    if model is None:

        st.warning("No trained model available.")

        return

    if mode == "Sample Images":

        sample_images_section()

    elif mode == "Upload Image":

        upload_mode(model)

    else:

        live_camera_mode(model)


if __name__ == "__main__":

    main()
