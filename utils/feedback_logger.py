"""
Feedback logging for the human-in-the-loop retraining loop.

Every time a user marks a prediction as correct/incorrect in the Streamlit
app, this module saves the image + metadata into feedback_data/, organized
so it can later be reviewed and folded back into the training dataset.

Folder layout produced:

feedback_data/
├── log.csv                     # one row per feedback event
├── correct/
│   └── <class_name>/<image>.jpg
└── incorrect/
    └── predicted_<class>__actual_<class>/<image>.jpg
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np

FEEDBACK_ROOT = Path("feedback_data")
LOG_PATH = FEEDBACK_ROOT / "log.csv"

LOG_FIELDS = [
    "timestamp",
    "image_filename",
    "predicted_class",
    "predicted_confidence",
    "was_correct",
    "corrected_class",
]


def _ensure_log_exists():
    FEEDBACK_ROOT.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()


def save_feedback(image: np.ndarray, predicted_class: str, confidence: float,
                   was_correct: bool, corrected_class: str = None) -> str:
    """
    Save an image + its feedback outcome.

    Returns the path where the image was saved.
    """
    _ensure_log_exists()

    timestamp = int(time.time())
    filename = f"{predicted_class}_{timestamp}.jpg"

    if was_correct:
        out_dir = FEEDBACK_ROOT / "correct" / predicted_class
    else:
        actual = corrected_class or "unknown"
        out_dir = FEEDBACK_ROOT / "incorrect" / f"predicted_{predicted_class}__actual_{actual}"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), image)

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writerow({
            "timestamp": timestamp,
            "image_filename": filename,
            "predicted_class": predicted_class,
            "predicted_confidence": round(confidence, 4) if confidence is not None else "",
            "was_correct": was_correct,
            "corrected_class": corrected_class or "",
        })

    return str(out_path)


def get_feedback_stats() -> dict:
    """Quick summary of accumulated feedback, used by the Streamlit sidebar."""
    _ensure_log_exists()

    total = correct = incorrect = 0
    with open(LOG_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row["was_correct"] == "True":
                correct += 1
            else:
                incorrect += 1

    return {"total": total, "correct": correct, "incorrect": incorrect}
