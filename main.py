"""
MAIN APPLICATION: Garment Print Defect Detection — Streamlit UI
FIXED VERSION FOR LIVE CAMERA
"""

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

from utils.feedback_logger import save_feedback, get_feedback_stats


# ============================================================================
# FIX WEBRTC / AIOICE CRASHES
# ============================================================================

os.environ["AIOICE_LOG_LEVEL"] = "CRITICAL"

logging.getLogger("aioice").setLevel(logging.CRITICAL)
logging.getLogger("aioice.ice").setLevel(logging.CRITICAL)
logging.getLogger("aiortc").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


def silence_known_errors(loop, context):
    """
    Ignore harmless WebRTC teardown errors.
    """

    exc = context.get("exception")
    message = context.get("message", "")

    if isinstance(exc, AttributeError):
        txt = str(exc)

        if "sendto" in txt:
            return

        if "call_exception_handler" in txt:
            return

    if "Task was destroyed but it is pending" in message:
        return

    loop.default_exception_handler(context)


def install_asyncio_handler():
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(silence_known_errors)
    except RuntimeError:
        pass


install_asyncio_handler()


# ============================================================================
# CONFIG
# ============================================================================

MODEL_PATH = "model/best.pt"

CONFIDENCE_THRESHOLD = 0.5

GOOD_CLASS_NAME = "good"

COLOR_GOOD = (0, 255, 0)
COLOR_DEFECT = (0, 0, 255)

# IMPORTANT:
# Empty ICE servers = avoids STUN retry crashes
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [],
        "iceTransportPolicy": "all",
    }
)

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


# ============================================================================
# MODEL
# ============================================================================

@st.cache_resource
def load_model():

    if not Path(MODEL_PATH).exists():
        return None

    return YOLO(MODEL_PATH)


# ============================================================================
# IMAGE HELPERS
# ============================================================================

def pil_to_cv2(img: Image.Image):

    arr = np.array(img.convert("RGB"))

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def run_inference(model, image_bgr):

    results = model(
        image_bgr,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
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

            name = class_names.get(cls_id, str(cls_id))

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

    return status, top_class, top_conf, annotated


def draw_status_bar(frame, status, top_class, conf, fps):

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

    color = COLOR_GOOD if status == "GOOD" else COLOR_DEFECT

    text = f"STATUS: {status}"

    if status == "DEFECTIVE":
        text += f" ({top_class})"

    if conf is not None:
        text += f"  {conf*100:.0f}%"

    cv2.putText(
        frame,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return frame


# ============================================================================
# VIDEO PROCESSOR
# ============================================================================

class DefectVideoProcessor:

    def __init__(self, model):

        self.model = model

        self.lock = threading.Lock()

        self.latest_frame_bgr = None

        self.latest_status = "GOOD"

        self.latest_class = GOOD_CLASS_NAME

        self.latest_conf = None

        self.prev_time = time.time()

        self.handler_installed = False

    def recv(self, frame):

        # Install asyncio handler inside WebRTC thread
        if not self.handler_installed:

            install_asyncio_handler()

            self.handler_installed = True

        img = frame.to_ndarray(format="bgr24")

        status, top_class, top_conf, annotated = run_inference(
            self.model,
            img,
        )

        now = time.time()

        fps = 1.0 / (now - self.prev_time)

        self.prev_time = now

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
            format="bgr24"
        )


# ============================================================================
# LIVE CAMERA MODE
# ============================================================================

def live_camera_mode(model):

    st.subheader("📷 Live Camera")

    st.caption(
        "Click START and allow browser camera permission."
    )

    ctx = webrtc_streamer(
        key="live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        video_processor_factory=lambda:
            DefectVideoProcessor(model),
        async_processing=True,
    )

    # IMPORTANT FIX
    if ctx.state.playing:

        st.success("Camera running")

    else:

        st.info("Click START to begin")

        return

    st.divider()

    if st.button("📸 Capture Current Frame"):

        if ctx.video_processor is None:

            st.warning("Camera not ready yet")

            return

        with ctx.video_processor.lock:

            frame = ctx.video_processor.latest_frame_bgr

            status = ctx.video_processor.latest_status

            top_class = ctx.video_processor.latest_class

            top_conf = ctx.video_processor.latest_conf

        if frame is None:

            st.warning("No frame captured yet")

            return

        st.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            caption="Captured Frame",
            use_container_width=True,
        )

        if status == "GOOD":

            st.success(
                f"STATUS: GOOD "
                f"({top_conf*100:.1f}%)"
            )

        else:

            st.error(
                f"STATUS: DEFECTIVE - {top_class} "
                f"({top_conf*100:.1f}%)"
            )


# ============================================================================
# MAIN
# ============================================================================

def main():

    st.set_page_config(
        page_title="Garment Print Defect Detection",
        layout="wide",
    )

    st.title("🧵 Garment Print Defect Detection")

    model = load_model()

    if model is None:

        st.error(f"Model not found: {MODEL_PATH}")

        return

    mode = st.radio(
        "Select Mode",
        [
            "Upload Image",
            "Live Camera",
        ],
        horizontal=True,
    )

    st.divider()

    if mode == "Live Camera":

        live_camera_mode(model)


if __name__ == "__main__":
    main()
