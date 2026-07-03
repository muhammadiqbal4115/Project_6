"""
Turn accumulated feedback (from the Streamlit app) into a batch of images
ready for re-labeling and retraining.

This is the "learning from feedback" loop:
  1. Users mark predictions correct/incorrect in the Streamlit app.
  2. Incorrect predictions + their corrected labels accumulate in feedback_data/.
  3. Run this script periodically to collect those images into a single
     folder you can upload to Roboflow (pre-sorted by corrected class, so
     labeling is much faster — just verify/adjust the box).
  4. Re-train with phase2_train_yolo.py using the expanded dataset.

Usage:
    python retrain_from_feedback.py
"""

import shutil
from pathlib import Path

FEEDBACK_ROOT = Path("feedback_data")
INCORRECT_DIR = FEEDBACK_ROOT / "incorrect"
BATCH_OUTPUT_DIR = Path("outputs/retraining_batch")


def collect_incorrect_feedback():
    if not INCORRECT_DIR.exists():
        print("[INFO] No feedback collected yet. Nothing to do.")
        return

    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for subfolder in INCORRECT_DIR.iterdir():
        if not subfolder.is_dir():
            continue

        # subfolder name looks like: predicted_X__actual_Y
        actual_class = subfolder.name.split("__actual_")[-1]
        class_out_dir = BATCH_OUTPUT_DIR / actual_class
        class_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in subfolder.glob("*.jpg"):
            shutil.copy(img_path, class_out_dir / img_path.name)
            count += 1

    print(f"[OK] Collected {count} misclassified images into {BATCH_OUTPUT_DIR}")


if __name__ == "__main__":
    collect_incorrect_feedback()
