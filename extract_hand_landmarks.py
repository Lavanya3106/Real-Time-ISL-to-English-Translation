"""
extract_hand_landmarks.py
Your ISL videos are close-ups of HANDS (no body, no face), so we use MediaPipe's
hand detector instead of the full-body one.

Per frame we keep, for each hand (left, right):
   21 points relative to the wrist, scaled by hand size   (63 numbers)
   where the wrist is in the picture                      (2 numbers)
   hand size                                              (1 number)
   "is this hand visible?" flag                           (1 number)
= 67 numbers per hand  ->  134 numbers per frame.

BEFORE RUNNING:
  1. pip install mediapipe opencv-python numpy
  2. Download this model file (once) and put it next to this script:
     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  3. Videos go in VIDEO_ROOT, one folder per word (accident_Cropped, call_Cropped, ...)

Run:  python extract_hand_landmarks.py
"""

import csv
import os
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")        # hides some harmless MediaPipe warning lines
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------- UPDATE THESE IF NEEDED ----------
VIDEO_ROOT = os.environ.get("VIDEO_ROOT", "isl_emergency")
MODEL_PATH = os.environ.get("MODEL_PATH", "hand_landmarker.task")
OUT_FOLDER = os.environ.get("OUT_FOLDER", "isl_hand_landmarks")
# --------------------------------------------

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}
PER_HAND = 67
FEAT_DIM = 2 * PER_HAND          # 134
MIN_HAND_SHARE = 0.20            # skip a clip if hands are seen in fewer than 20% of frames


def label_of(video_path):
    """Folder name like 'accident_Cropped' -> 'accident'."""
    rel = video_path.relative_to(VIDEO_ROOT)
    name = rel.parts[0] if len(rel.parts) > 1 else video_path.stem
    return name.lower().replace("_cropped", "").replace("-cropped", "").strip()


def signer_take_of(video_path):
    """'accident_Crop_006_01.avi' -> ('006', '01')."""
    m = re.search(r"_Crop_(\d+)_(\d+)", video_path.name)
    return (m.group(1), m.group(2)) if m else ("", "")


def one_hand_features(landmarks):
    """21 landmarks -> 67 numbers (zeros if the hand is unusable)."""
    pts = np.array([(p.x, p.y, p.z) for p in landmarks[:21]], dtype=np.float32)
    if len(pts) < 21:
        return np.zeros(PER_HAND, dtype=np.float32)
    wrist = pts[0]
    size = float(np.linalg.norm(pts[9, :2] - pts[0, :2]))     # wrist -> base of middle finger
    if size < 1e-4:
        return np.zeros(PER_HAND, dtype=np.float32)
    rel = np.clip((pts - wrist) / size, -10, 10)
    return np.concatenate([rel.reshape(-1), wrist[:2], [size], [1.0]]).astype(np.float32)


def result_to_frame(result):
    """One MediaPipe result -> 134 numbers. Slot 0 = 'Left' hand, slot 1 = 'Right' hand."""
    out = np.zeros(FEAT_DIM, dtype=np.float32)
    used = [False, False]
    for lms, handed in zip(result.hand_landmarks, result.handedness):
        name = handed[0].category_name if handed else "Right"
        k = 0 if name == "Left" else 1
        if used[k]:                    # both hands got the same label -> use the other slot
            k = 1 - k
            if used[k]:
                continue
        used[k] = True
        out[k * PER_HAND:(k + 1) * PER_HAND] = one_hand_features(lms)
    return out


def read_video_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(i * 1000 / fps)
        i += 1
    cap.release()


def process_video(mp, options, path):
    """Returns (clip array (T,134) float16, share of frames with a hand)."""
    frames = []
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:  # fresh tracker per video
        for rgb, ts in read_video_frames(path):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frames.append(result_to_frame(landmarker.detect_for_video(img, ts)))
    if not frames:
        return None, 0.0
    arr = np.array(frames, dtype=np.float32)
    seen = (arr[:, PER_HAND - 1] > 0) | (arr[:, FEAT_DIM - 1] > 0)    # visible flags
    return arr.astype(np.float16), float(seen.mean())


def main():
    import mediapipe as mp

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model file not found: {MODEL_PATH}\n"
                         "Download it from the link at the top of this script and put it next to the script.")
    videos = sorted(p for p in Path(VIDEO_ROOT).rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"No videos found inside '{VIDEO_ROOT}'. Check the folder name.")
    print(f"Found {len(videos)} videos")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3)

    # ---- quick check first: one video per word, so we don't waste time if detection is bad ----
    first_per_word = {}
    for v in videos:
        first_per_word.setdefault(label_of(v), v)
    print("\nQuick check (one video per word) - share of frames where a hand was found:")
    shares = []
    for word, v in first_per_word.items():
        _, share = process_video(mp, options, v)
        shares.append(share)
        print(f"  {word:10} {100 * share:5.0f}%")
    avg = sum(shares) / len(shares)
    if avg < 0.5:
        raise SystemExit(f"\nHands were found in only {100 * avg:.0f}% of frames, so I stopped. "
                         "Send me this output and I'll fix it before you wait for the full run.")
    print(f"Looks good ({100 * avg:.0f}% on average). Processing everything now...\n")

    os.makedirs(OUT_FOLDER, exist_ok=True)
    rows, skipped, skipped_by_word = [], 0, Counter()
    for n, vp in enumerate(videos, 1):
        label = label_of(vp)
        signer, take = signer_take_of(vp)
        out_dir = Path(OUT_FOLDER) / label
        out_dir.mkdir(parents=True, exist_ok=True)
        rel_parts = vp.relative_to(VIDEO_ROOT).with_suffix("").parts
        uid = "_".join(rel_parts[1:]) if len(rel_parts) > 1 else vp.stem
        out_path = out_dir / (uid + ".npy")

        if out_path.exists():                                   # finished earlier, lets you re-run safely
            clip = np.load(out_path)
            seen = (clip[:, PER_HAND - 1] > 0) | (clip[:, FEAT_DIM - 1] > 0)
            share = float(seen.mean())
        else:
            clip, share = process_video(mp, options, vp)
            if clip is None or share < MIN_HAND_SHARE or len(clip) < 5:
                skipped += 1
                skipped_by_word[label] += 1
                print(f"  skipped (hands not found): {vp.name}")
                continue
            np.save(out_path, clip)
        rows.append({"file": f"{label}/{out_path.name}", "label": label, "signer": signer, "take": take,
                     "num_frames": len(clip), "video": vp.name, "hands_seen": f"{share:.2f}"})
        if n % 25 == 0:
            print(f"  {n}/{len(videos)} videos done...")

    with open(os.path.join(OUT_FOLDER, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "signer", "take", "num_frames", "video", "hands_seen"])
        w.writeheader()
        w.writerows(rows)

    labels = sorted({r["label"] for r in rows})
    print(f"\nDone. {len(rows)} clips saved, {skipped} skipped.")
    print("Words found:", ", ".join(labels))
    if skipped:
        print("Skipped per word:", dict(skipped_by_word))
    for lab in labels:
        print(f"  {lab}: {sum(r['label'] == lab for r in rows)} clips")
    seen_all = [float(r["hands_seen"]) for r in rows]
    if seen_all:
        print(f"Average share of frames where a hand was detected: {100 * sum(seen_all) / len(seen_all):.0f}%")
    print(f"Different signers: {len({r['signer'] for r in rows if r['signer']})}")
    print(f"Index file: {os.path.join(OUT_FOLDER, 'index.csv')}")


if __name__ == "__main__":
    main()