"""
extract_include.py
Turns the INCLUDE Indian Sign Language videos (AI4Bharat, 263 words) into the SAME 134-number-per-frame
hand features your current project already uses, so app.py keeps working.

INCLUDE folder layout (after unzipping the parts you downloaded):
    include/<Category>/<N. word>/<video files>      e.g. include/Animals/1. Dog/MVI_5177.MOV
The word is taken from the folder that directly contains the video ("1. Dog" -> "dog").

BEFORE RUNNING
  1. pip install mediapipe opencv-python numpy
  2. hand_landmarker.task and extract_hand_landmarks.py must be in this folder
  3. Put the unzipped INCLUDE folders inside ./include   (or set INCLUDE_ROOT)

CHOOSING WORDS (pick one way)
  python extract_include.py                       -> the 50 words that have the most videos (N_WORDS=50)
  N_WORDS=30 python extract_include.py            -> top 30 words
  WORDS="dog,cat,hello,water" python extract_include.py   -> exactly these words

Safe to stop and re-run: finished videos are skipped.
"""

import csv
import os
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

from extract_hand_landmarks import FEAT_DIM, PER_HAND, VIDEO_EXTS, result_to_frame

# ---------- SETTINGS ----------
INCLUDE_ROOT = os.environ.get("INCLUDE_ROOT", "include")
MODEL_PATH = os.environ.get("MODEL_PATH", "hand_landmarker.task")
OUT_FOLDER = os.environ.get("OUT_FOLDER", "isl_include_landmarks")
N_WORDS = int(os.environ.get("N_WORDS", 80))
WORDS = [w.strip().lower() for w in os.environ.get("WORDS", "").split(",") if w.strip()]
MIN_CLIPS_PER_WORD = int(os.environ.get("MIN_CLIPS", 8))   # words with fewer clips are ignored
MAX_SIDE = 1280            # INCLUDE is 1920x1080; shrinking a bit keeps it fast but hands stay big enough
MIN_HAND_SHARE = 0.20      # skip a video if hands are seen in fewer than 20% of frames
# ------------------------------


def label_of(path):
    folder = path.parent
    while folder.name.strip().lower() in {"extra", "extras"} and folder.parent != folder:
        folder = folder.parent                                          # ".../5. Beautiful/Extra/x.mov" -> Beautiful
    name = re.sub(r"^\s*\d+\s*[.\-_)]*\s*", "", folder.name)           # "12. Good Morning" -> "Good Morning"
    return re.sub(r"\s+", "_", name.strip().lower())


def process_video(mp, options, path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps == fps and fps >= 1 else 30.0
    frames, i = [], 0
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as lm:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            if max(h, w) > MAX_SIDE:
                s = MAX_SIDE / max(h, w)
                bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            frames.append(result_to_frame(lm.detect_for_video(img, int(i * 1000 / fps))))
            i += 1
    cap.release()
    if not frames:
        return None, 0.0
    arr = np.array(frames, dtype=np.float32)
    seen = (arr[:, PER_HAND - 1] > 0) | (arr[:, FEAT_DIM - 1] > 0)
    return arr.astype(np.float16), float(seen.mean())


def main():
    import mediapipe as mp

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model file not found: {MODEL_PATH}")
    videos = sorted(p for p in Path(INCLUDE_ROOT).rglob("*") if p.suffix.lower() in VIDEO_EXTS | {".mov"})
    if not videos:
        raise SystemExit(f"No videos found inside '{INCLUDE_ROOT}'. Unzip the INCLUDE parts there.")
    by_word = defaultdict(list)
    for v in videos:
        by_word[label_of(v)].append(v)
    print(f"Found {len(videos)} videos, {len(by_word)} different words")

    if WORDS:
        chosen = [w for w in WORDS if w in by_word]
        missing = [w for w in WORDS if w not in by_word]
        if missing:
            print("Not found (check spelling / did you download that part?):", ", ".join(missing))
    else:
        ok_words = [(w, len(v)) for w, v in by_word.items() if len(v) >= MIN_CLIPS_PER_WORD]
        ok_words.sort(key=lambda t: (-t[1], t[0]))
        chosen = sorted(w for w, _ in ok_words[:N_WORDS])
    if len(chosen) < 2:
        raise SystemExit("Need at least 2 words to train.")
    print(f"Using {len(chosen)} words: {', '.join(chosen)}\n")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3)

    os.makedirs(OUT_FOLDER, exist_ok=True)
    todo = [(w, v) for w in chosen for v in by_word[w]]
    rows, skipped = [], Counter()
    for n, (word, vp) in enumerate(todo, 1):
        rel = vp.relative_to(INCLUDE_ROOT).as_posix()
        uid = f"{vp.stem}_{zlib.crc32(rel.encode()):08x}"
        out_dir = Path(OUT_FOLDER) / word
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (uid + ".npy")
        if out_path.exists():
            clip = np.load(out_path)
            seen = (clip[:, PER_HAND - 1] > 0) | (clip[:, FEAT_DIM - 1] > 0)
            share = float(seen.mean())
        else:
            clip, share = process_video(mp, options, vp)
            if clip is None or share < MIN_HAND_SHARE or len(clip) < 5:
                skipped[word] += 1
                continue
            np.save(out_path, clip)
        rows.append({"file": f"{word}/{out_path.name}", "label": word, "num_frames": len(clip),
                     "video": vp.name, "relpath": rel, "hands_seen": f"{share:.2f}"})
        if n % 25 == 0:
            print(f"  {n}/{len(todo)} videos done...")

    with open(os.path.join(OUT_FOLDER, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "num_frames", "video", "relpath", "hands_seen"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. {len(rows)} clips saved, {sum(skipped.values())} skipped (hands not found).")
    print("Clips and average hand-detection per word (low % = hard word for the hand detector):")
    for word in chosen:
        r = [float(x["hands_seen"]) for x in rows if x["label"] == word]
        if r:
            print(f"  {word:18} {len(r):3} clips   hands seen {100 * np.mean(r):3.0f}%")
    print(f"Index file: {os.path.join(OUT_FOLDER, 'index.csv')}")


if __name__ == "__main__":
    main()