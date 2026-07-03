"""
MAIN APPLICATION: Garment Print Defect Detection — Streamlit UI

Two modes:
  1. Upload Image — upload a photo, get a defect prediction.
  2. Live Camera — opens your browser's camera (works on phone or laptop,
     no DroidCam needed) and runs detection on the live video feed.

Both modes support feedback (correct/incorrect), which is logged and used
to build a retraining queue (human-in-the-loop / feedback-driven learning).

Usage:
    streamlit run streamlit_app.py

Note: Live Camera mode requires the browser to grant camera permission.
On phones, open the Streamlit app URL directly in the phone's browser.
"""

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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

from utils.feedback_logger import save_feedback, get_feedback_stats

# ---------------------------------------------------------------------------
# Silence known aioice/aiortc teardown noise.
#
# When the WebRTC camera session stops (button press, tab close, or a
# Streamlit rerun tearing down the video thread), two harmless but noisy
# things happen during aiortc/aioice cleanup:
#
# 1. A pending STUN retransmission timer (aioice Transaction.__retry) can
#    fire AFTER the transport/event loop are already closed, raising:
#        AttributeError: 'NoneType' object has no attribute 'sendto'
#        AttributeError: 'NoneType' object has no attribute 'call_exception_handler'
#    This goes through the event loop's exception handler, so we can filter
#    it there (handled by _silence_known_teardown_errors below).
#
# 2. aiortc's own internal housekeeping tasks (RTCIceTransport._monitor,
#    Connection.query_consent, RTCPeerConnection.close) sometimes get
#    garbage-collected while still pending, because the background thread's
#    event loop is torn down before they finish. This prints directly via
#    Task.__del__ as "Task was destroyed but it is pending!" — it does NOT
#    go through the exception handler, so we filter it at the stderr level
#    instead (handled by _FilteredStderr below).
#
# Both are shutdown-ordering issues inside aiortc/aioice themselves, not in
# this app. Detection and the UI are unaffected either way — this just keeps
# the terminal clean.
# ---------------------------------------------------------------------------
logging.getLogger("aioice.ice").setLevel(logging.CRITICAL)
logging.getLogger("aioice").setLevel(logging.CRITICAL)
logging.getLogger("aiortc").setLevel(logging.CRITICAL)


def _silence_known_teardown_errors(loop, context):
    """Custom asyncio exception handler: drop the known aioice teardown
    AttributeErrors, forward anything else to the default handler."""
    exc = context.get("exception")
    message = context.get("message", "")
    if isinstance(exc, AttributeError) and "NoneType" in str(exc):
        return
    if "sendto" in message or "call_exception_handler" in message:
        return
    loop.default_exception_handler(context)


def _install_exception_handler_on_current_loop():
    """Attach the custom handler to whichever event loop is running in the
    calling thread. streamlit-webrtc runs its WebRTC session on its own
    background thread with its own event loop, so this is called both from
    the main thread (best-effort) and from inside the video processor the
    first time it runs (which is where it actually matters)."""
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(_silence_known_teardown_errors)
    except RuntimeError:
        pass


_KNOWN_NOISY_SNIPPETS = (
    "Task was destroyed but it is pending",
    "RTCIceTransport._monitor",
    "query_consent",
    "RTCPeerConnection.close",
    "task: <Task pending name=",
)


class _FilteredStderr:
    """Wraps sys.stderr and drops the specific multi-line 'Task was
    destroyed but it is pending!' blocks that aiortc emits directly via
    Task.__del__ during WebRTC session teardown. Everything else passes
    through untouched so real errors are still visible."""

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._suppress_until_blank = False

    def write(self, text):
        if any(snippet in text for snippet in _KNOWN_NOISY_SNIPPETS):
            self._suppress_until_blank = True
            return
        if self._suppress_until_blank:
            # Continuation lines of the same traceback (indented "File ..."
            # lines, coro=... wait_for=... etc.) — keep dropping until we
            # hit a line that clearly starts a new, unrelated message.
            if text.strip() == "" or text.startswith("  ") or text.lstrip().startswith(("File ", "task:")):
                return
            self._suppress_until_blank = False
        self._wrapped.write(text)

    def flush(self):
        self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


if not isinstance(sys.stderr, _FilteredStderr):
    sys.stderr = _FilteredStderr(sys.stderr)

_install_exception_handler_on_current_loop()

MODEL_PATH = "model/best.pt"
CONFIDENCE_THRESHOLD = 0.5
GOOD_CLASS_NAME = "good"
SAMPLE_IMAGES_DIR = Path("sample_images")

SAMPLE_IMAGES = [
    {"file": "sample1.jpg", "caption": "Delik"},
    {"file": "sample2.jpg", "caption": "Delik"},
    {"file": "sample3.jpg", "caption": "Jut"},
    {"file": "sample4.jpg", "caption": "Leke"},
]

def sample_images_section():
    st.subheader("📷 Sample Images")
    st.caption("Download these sample images to test the app.")

    cols = st.columns(4)

    for col, img in zip(cols, SAMPLE_IMAGES):
        img_path = SAMPLE_IMAGES_DIR / img["file"]

        with col:
            if img_path.exists():
                st.image(str(img_path), caption=img["caption"], width='stretch')

                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                st.download_button(
                    label="⬇️ Download",
                    data=img_bytes,
                    file_name=img["file"],
                    mime="image/jpeg",
                    key=f"download_{img['file']}",
                    width='stretch',
                )
            else:
                st.warning(f"Missing: {img['file']}")

DEFECT_CLASSES = [
    "good",
    'Delik', 
    'Jut', 
    'Leke',
    "pinhole",
    "blur",
    "misregistration",
    "ink_bleed",
    "ghosting",
    "uneven_deposit",
    "low_opacity",
]

COLOR_GOOD = (0, 200, 0)
COLOR_DEFECT = (0, 0, 255)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Garment Print Defect Detection", layout="wide")



@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        return None
    return YOLO(MODEL_PATH)


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def run_inference(model, image_bgr: np.ndarray):
    results = model(image_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)
    annotated = results[0].plot()  # BGR
    boxes = results[0].boxes
    class_names = results[0].names

    status = "GOOD"
    top_class = GOOD_CLASS_NAME
    top_conf = None

    if boxes is not None and len(boxes) > 0:
        confs, names = [], []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = class_names.get(cls_id, str(cls_id))
            confs.append(conf)
            names.append(name)

        best_idx = int(np.argmax(confs))
        top_class = names[best_idx]
        top_conf = confs[best_idx]
        status = "GOOD" if top_class.lower() == GOOD_CLASS_NAME else "DEFECTIVE"

    return status, top_class, top_conf, annotated


def draw_status_bar(frame, status, top_class, conf, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    color = COLOR_GOOD if status == "GOOD" else COLOR_DEFECT
    text = f"STATUS: {status}"
    if status == "DEFECTIVE":
        text += f" ({top_class})"
    if conf is not None:
        text += f"  {conf*100:.0f}%"

    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


class DefectVideoProcessor:
    """
    Runs YOLO inference on each incoming video frame in Live Camera mode.
    Keeps the latest frame + prediction in thread-safe state so the main
    Streamlit thread can read it (e.g. for the feedback buttons).
    """

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
        # Install the exception handler on THIS thread's event loop the
        # first time a frame is processed — this is the loop that actually
        # owns the ICE/STUN transport, so this is where it matters most.
        if not self._handler_installed:
            _install_exception_handler_on_current_loop()
            self._handler_installed = True

        img = frame.to_ndarray(format="bgr24")

        status, top_class, top_conf, annotated = run_inference(self.model, img)

        now = time.time()
        fps = 1.0 / (now - self._prev_time) if now != self._prev_time else 0.0
        self._prev_time = now

        annotated = draw_status_bar(annotated, status, top_class, top_conf, fps)

        with self.lock:
            self.latest_frame_bgr = img.copy()
            self.latest_status = status
            self.latest_class = top_class
            self.latest_conf = top_conf

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def render_feedback_controls(image_bgr, top_class, top_conf, key_prefix):
    """Shared feedback UI used by both Upload and Live Camera modes."""
    state_key = f"feedback_done_{key_prefix}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    if st.session_state[state_key]:
        st.info("Thanks — feedback recorded.")
        if st.button("Give feedback on another capture", key=f"reset_{key_prefix}"):
            st.session_state[state_key] = False
            st.rerun()
        return

    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 2])

    with fb_col1:
        if st.button("👍 Correct", width='stretch', key=f"correct_{key_prefix}"):
            save_feedback(image_bgr, top_class, top_conf, was_correct=True)
            st.session_state[state_key] = True
            st.rerun()

    with fb_col2:
        wrong_clicked = st.button("👎 Incorrect", width='stretch', key=f"incorrect_{key_prefix}")

    with fb_col3:
        if wrong_clicked:
            st.session_state[f"show_correction_{key_prefix}"] = True

        if st.session_state.get(f"show_correction_{key_prefix}"):
            corrected = st.selectbox(
                "What's the correct label?", DEFECT_CLASSES, key=f"correction_{key_prefix}"
            )
            if st.button("Submit correction", key=f"submit_{key_prefix}"):
                save_feedback(image_bgr, top_class, top_conf, was_correct=False, corrected_class=corrected)
                st.session_state[state_key] = True
                st.rerun()


def upload_mode(model):
    uploaded_file = st.file_uploader("Upload shirt print image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload an image to run detection.")
        return

    image = Image.open(uploaded_file)
    image_bgr = pil_to_cv2(image)

    with st.spinner("Running detection..."):
        status, top_class, top_conf, annotated_bgr = run_inference(model, image_bgr)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Original")
        st.image(image, width='stretch')
    with col_b:
        st.subheader("Detection Result")
        st.image(annotated_rgb, width='stretch')

    st.divider()

    if status == "GOOD":
        st.success("STATUS: GOOD" + (f"  ({top_conf*100:.1f}% confidence)" if top_conf else ""))
    else:
        st.error(f"STATUS: DEFECTIVE — {top_class}" + (f"  ({top_conf*100:.1f}% confidence)" if top_conf else ""))

    st.subheader("Was this prediction correct?")
    render_feedback_controls(image_bgr, top_class, top_conf, key_prefix=f"{uploaded_file.name}_{uploaded_file.size}")


def live_camera_mode(model):
    st.caption(
        "Click **Start** and allow camera access in your browser. "
        "On a phone, open this app's URL directly in your phone's browser."
    )

    ctx = webrtc_streamer(
        key="defect-detection-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: DefectVideoProcessor(model),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.divider()
    st.subheader("Capture current frame for feedback")

    if ctx.video_processor is None:
        st.info("Start the camera above, then you can capture a frame to give feedback on.")
        return

    if st.button("📸 Capture current frame"):
        with ctx.video_processor.lock:
            frame = ctx.video_processor.latest_frame_bgr
            status = ctx.video_processor.latest_status
            top_class = ctx.video_processor.latest_class
            top_conf = ctx.video_processor.latest_conf

        if frame is None:
            st.warning("No frame captured yet — make sure the camera is running.")
        else:
            st.session_state["captured_frame"] = frame
            st.session_state["captured_status"] = status
            st.session_state["captured_class"] = top_class
            st.session_state["captured_conf"] = top_conf
            st.session_state["capture_id"] = st.session_state.get("capture_id", 0) + 1

    if "captured_frame" in st.session_state:
        frame = st.session_state["captured_frame"]
        status = st.session_state["captured_status"]
        top_class = st.session_state["captured_class"]
        top_conf = st.session_state["captured_conf"]

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured frame", width='stretch')

        if status == "GOOD":
            st.success("STATUS: GOOD" + (f"  ({top_conf*100:.1f}% confidence)" if top_conf else ""))
        else:
            st.error(f"STATUS: DEFECTIVE — {top_class}" + (f"  ({top_conf*100:.1f}% confidence)" if top_conf else ""))

        render_feedback_controls(
            frame, top_class, top_conf, key_prefix=f"live_{st.session_state.get('capture_id', 0)}"
        )


def main():
    st.title("🧵 Garment Print Defect Detection")

    model = load_model()

    with st.sidebar:
        st.header("Model Status")
        if model is None:
            st.error(f"No trained model found at `{MODEL_PATH}`.")
            st.info("Train one with phase2_train_yolo.py and place best.pt there.")
        else:
            st.success("Model loaded.")

        st.header("Feedback Stats")
        stats = get_feedback_stats()
        st.metric("Total feedback", stats["total"])
        col1, col2 = st.columns(2)
        col1.metric("Correct", stats["correct"])
        col2.metric("Incorrect", stats["incorrect"])

        st.header("Retraining")
        if st.button("🔁 Run retrain_from_feedback.py", width='stretch'):
            with st.spinner("Collecting feedback into retraining batch..."):
                result = subprocess.run(
                    ["python", "retrain_from_feedback.py"],
                    capture_output=True, text=True,
                )
            if result.returncode == 0:
                st.success("Done. See output below.")
            else:
                st.error("Script exited with an error. See output below.")
            st.code(result.stdout + result.stderr or "(no output)")

        st.header("Rerun")
        rerun = st.button("Click Me")
        if rerun:
            st.rerun()


    mode = st.radio("Modes", ["Sample Images", "Upload Image", "Live Camera"], horizontal=True)
    st.divider()

    if model is None:
        st.warning("Can't run detection — no trained model available yet.")
        return
    if mode == "Sample Images":
        sample_images_section()
    elif mode == "Upload Image":
        upload_mode(model)
    else:
        live_camera_mode(model)


if __name__ == "__main__":
    main()