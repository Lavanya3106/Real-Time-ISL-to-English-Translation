"""
demo.py - Healthcare Sign Translator (desktop demo, version 2)

Two kinds of signs, two modes (press S to switch):
  MOVING signs  (video)    8 words : accident, call, doctor, help, hot, lose, pain, thief
  STILL signs   (pictures) 20 words: afraid, agree, assistance, bad, become, college, doctor, from,
                                     pain, pray, secondary, skin, small, specific, stand, today,
                                     warn, which, work, you
Both modes: the hand skeleton is drawn, the trained model recognizes the sign, and you collect
signs into an English sentence. Everything shown comes from people the model NEVER saw in training.

Keys (click on the demo window first):
  S          switch between moving signs and still signs
  MOVING:    1-8 pick a word | N/P next/previous clip | W record 3 s from your webcam
  STILL:     N/P next/previous word | R show different pictures of the same word
  SPACE add the recognized word to the sentence | BACKSPACE undo | C clear | Q or ESC quit

Files needed in the SAME folder:
  demo.py, extract_hand_landmarks.py, extract_static_landmarks.py, hand_landmarker.task
  moving signs: isl_model.pt, isl_test_clips.json, folder isl_emergency
  still signs : isl_static_model.pt, isl_static_test_images.json, folder isl_static
  (if the still-sign files are missing, the demo simply runs with the moving signs only)

Run:  python demo.py
"""

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import torch
import torch.nn as nn

from extract_hand_landmarks import PER_HAND, result_to_frame      # same folder
from extract_static_landmarks import MIN_SIDE, static_features    # same folder

VIDEO_ROOT = os.environ.get("VIDEO_ROOT", "isl_emergency")
IMG_ROOT = os.environ.get("IMG_ROOT", "isl_static")
HAND_MODEL = os.environ.get("MODEL_PATH", "hand_landmarker.task")
CLF_MODEL, TEST_LIST = "isl_model.pt", "isl_test_clips.json"
STATIC_MODEL, STATIC_LIST = "isl_static_model.pt", "isl_static_test_images.json"
CONF_MIN_MOVING = 0.50    # below this the app says "not sure" and refuses to add the word
CONF_MIN_STILL = 0.40
PICTURES_PER_WORD = 8     # still signs: how many pictures of the sign are averaged
REC_SECONDS = 3
VIDEO_W, VIDEO_H, PANEL_W = 640, 500, 450
TITLE = "Healthcare Sign Translator"

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

# ---------------------------------------------------------------
# Sentence builder (simple rules)
# ---------------------------------------------------------------
SINGLE = {
    # moving signs
    "pain": "I am in pain.",
    "hot": "I feel very hot.",
    "doctor": "I need a doctor.",
    "help": "Please help me.",
    "call": "Please make a call.",
    "accident": "There has been an accident.",
    "lose": "I have lost something.",
    "thief": "There is a thief here.",
    # still signs (the rest simply show the word)
    "afraid": "I am afraid.",
    "agree": "I agree.",
    "assistance": "I need assistance.",
    "bad": "I feel bad.",
    "skin": "I have a problem with my skin.",
    "pray": "I want to pray.",
    "warn": "Warning, please be careful.",
    "small": "It is small.",
}
PAIRS = {
    frozenset(("call", "doctor")): "Please call a doctor.",
    frozenset(("call", "help")): "Please call for help.",
    frozenset(("help", "doctor")): "Please help me, I need a doctor.",
    frozenset(("assistance", "doctor")): "I need assistance from a doctor.",
    frozenset(("bad", "pain")): "I feel bad and I am in pain.",
}


def build_sentence(words):
    words = [w for i, w in enumerate(words) if i == 0 or w != words[i - 1]]   # same sign twice in a row = once
    out, i = [], 0
    while i < len(words):
        pair = frozenset(words[i:i + 2])
        if i + 1 < len(words) and len(pair) == 2 and pair in PAIRS:
            out.append(PAIRS[pair])
            i += 2
        else:
            out.append(SINGLE.get(words[i], words[i].capitalize() + "."))
            i += 1
    return " ".join(out)


# ---------------------------------------------------------------
# Recognition models (same designs as in the two training scripts)
# ---------------------------------------------------------------
class SignGRU(nn.Module):
    def __init__(self, n_classes, d_in, hidden=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(hidden * 4, n_classes))

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.head(torch.cat([h.mean(1), h.max(1).values], dim=1))


class MLP(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


def resample(x, n):
    t = len(x)
    pos = np.linspace(0, t - 1, n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (pos - lo)[:, None]
    return (x[lo] * (1 - w) + x[hi] * w).astype(np.float32)


class MovingRecognizer:
    def __init__(self):
        ck = torch.load(CLF_MODEL, map_location="cpu")
        self.labels, self.fixed_len = ck["labels"], ck["fixed_len"]
        self.model = SignGRU(len(self.labels), ck["d_in"])
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, feats):
        x = torch.tensor(resample(feats, self.fixed_len))[None]
        return torch.softmax(self.model(x), dim=1)[0].numpy()


class StillRecognizer:
    def __init__(self):
        ck = torch.load(STATIC_MODEL, map_location="cpu")
        self.labels = ck["labels"]
        self.model = MLP(ck["d_in"], len(self.labels))
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, feats):                       # several pictures -> average of their answers
        return torch.softmax(self.model(torch.tensor(feats)), dim=1).mean(0).numpy()


# ---------------------------------------------------------------
# Finding hands + predicting
# ---------------------------------------------------------------
def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps == fps and fps >= 1 else 30.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    return frames, fps


def hands_to_pixels(res, w, h):
    return [[(p.x * w, p.y * h) for p in lms[:21]] for lms in res.hand_landmarks]


def analyze_moving(mp, options, recognizer, frames, fps, name, truth=""):
    feats, hands_px = [], []
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        for i, bgr in enumerate(frames):
            h, w = bgr.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(img, int(i * 1000 / fps))
            feats.append(result_to_frame(res))
            hands_px.append(hands_to_pixels(res, w, h))
    feats = np.array(feats, dtype=np.float32)
    seen = float(((feats[:, PER_HAND - 1] > 0) | (feats[:, 2 * PER_HAND - 1] > 0)).mean()) if len(feats) else 0.0
    probs = recognizer.predict(feats) if len(feats) >= 5 and seen >= 0.2 else None
    return {"name": name, "truth": truth, "frames": frames, "fps": fps, "hands": hands_px, "probs": probs,
            "seen": seen, "labels": recognizer.labels, "conf_min": CONF_MIN_MOVING}


def analyze_still(mp, landmarker, recognizer, paths, name, truth=""):
    frames, hands_px, feats = [], [], []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        if min(h, w) < MIN_SIDE:                                  # same enlarging as during training
            s = MIN_SIDE / min(h, w)
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
            h, w = bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = landmarker.detect(img)
        v = result_to_frame(res)
        frames.append(bgr)
        hands_px.append(hands_to_pixels(res, w, h))
        if v[PER_HAND - 1] > 0 or v[2 * PER_HAND - 1] > 0:
            feats.append(static_features(v))
    seen = len(feats) / max(1, len(frames))
    probs = recognizer.predict(np.array(feats, dtype=np.float32)) if len(feats) >= 2 else None
    return {"name": name, "truth": truth, "frames": frames, "fps": 3.0, "hands": hands_px, "probs": probs,
            "seen": seen, "labels": recognizer.labels, "conf_min": CONF_MIN_STILL}


# ---------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------
def put(img, text, x, y, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def wrap(text, max_px, scale=0.6):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > max_px and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def result_of(a):
    """-> (word or None, confidence, text for the big banner, color)"""
    if a is None:
        return None, 0.0, "", (255, 255, 255)
    if a["probs"] is None:
        return None, 0.0, "NO HANDS FOUND", (80, 80, 255)
    k = int(np.argmax(a["probs"]))
    p = float(a["probs"][k])
    word = a["labels"][k]
    if p < a["conf_min"]:
        return None, p, "NOT SURE", (0, 200, 255)
    return word, p, f"{word.upper()}  {100 * p:.0f}%", (120, 255, 120)


def render(a, frame_idx, words, message="", status="", mode="moving"):
    canvas = np.full((VIDEO_H, VIDEO_W + PANEL_W, 3), 30, dtype=np.uint8)
    if a is not None and a["frames"]:
        i = frame_idx % len(a["frames"])
        bgr = a["frames"][i].copy()
        for k, hand in enumerate(a["hands"][i]):
            color = (80, 220, 80) if k == 0 else (60, 160, 255)
            for p, q in BONES:
                cv2.line(bgr, tuple(int(v) for v in hand[p]), tuple(int(v) for v in hand[q]), color, 2, cv2.LINE_AA)
            for pt in hand:
                cv2.circle(bgr, (int(pt[0]), int(pt[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)
        h, w = bgr.shape[:2]
        s = min(VIDEO_W / w, VIDEO_H / h)
        small = cv2.resize(bgr, (int(w * s), int(h * s)))
        canvas[:small.shape[0], :small.shape[1]] = small
    else:
        put(canvas, status or "Press N to start", 30, VIDEO_H // 2, 0.8)

    x0 = VIDEO_W + 20
    put(canvas, "HEALTHCARE SIGN TRANSLATOR", x0, 32, 0.7, (255, 255, 255), 2)
    sub = "MOVING signs (video) - 8 words" if mode == "moving" else "STILL signs (pictures) - 20 words"
    put(canvas, f"ISL | {sub}", x0, 56, 0.45, (180, 180, 180))
    cv2.line(canvas, (x0, 68), (x0 + PANEL_W - 40, 68), (90, 90, 90), 1)

    word, p, banner, color = result_of(a)
    if a is not None:
        put(canvas, f"{'Clip' if mode == 'moving' else 'Pictures'}: {a['name']}", x0, 92, 0.45, (180, 180, 180))
        if a["truth"]:
            ok = "" if word is None else ("  (correct)" if word == a["truth"] else "  (wrong)")
            put(canvas, f"Real word: {a['truth']}{ok}", x0, 112, 0.45, (180, 180, 180))
    put(canvas, "I RECOGNIZE:", x0, 150, 0.55, (200, 200, 200))
    put(canvas, banner or "-", x0, 185, 0.9, color, 2)
    if a is not None and a["probs"] is not None:
        for r, k in enumerate(np.argsort(-a["probs"])[:3]):
            y = 215 + r * 24
            put(canvas, f"{a['labels'][k]:11}", x0, y, 0.5)
            cv2.rectangle(canvas, (x0 + 110, y - 13), (x0 + 110 + int(200 * a["probs"][k]), y + 2), (120, 200, 120), -1)
            put(canvas, f"{100 * a['probs'][k]:.0f}%", x0 + 325, y, 0.5)

    cv2.line(canvas, (x0, 295), (x0 + PANEL_W - 40, 295), (90, 90, 90), 1)
    put(canvas, "SIGNS SO FAR:", x0, 320, 0.55, (200, 200, 200))
    put(canvas, " > ".join(words) if words else "(none yet - press SPACE to add)", x0, 345, 0.55, (255, 220, 120))
    put(canvas, "SENTENCE:", x0, 380, 0.55, (200, 200, 200))
    for i, line in enumerate(wrap(build_sentence(words), PANEL_W - 40, 0.65)[:3] if words else ["-"]):
        put(canvas, line, x0, 408 + i * 26, 0.65, (255, 255, 255), 1)
    if mode == "moving":
        put(canvas, "S: still signs   1-8: pick a word   N/P: clip   W: webcam", x0, VIDEO_H - 32, 0.4, (150, 150, 150))
    else:
        put(canvas, "S: moving signs   N/P: next/previous word   R: other pictures", x0, VIDEO_H - 32, 0.4, (150, 150, 150))
    put(canvas, "SPACE: add word   BKSP: undo   C: clear   Q: quit", x0, VIDEO_H - 12, 0.4, (150, 150, 150))
    if message:
        cv2.rectangle(canvas, (0, VIDEO_H - 34), (VIDEO_W, VIDEO_H), (0, 0, 0), -1)
        put(canvas, message, 12, VIDEO_H - 11, 0.6, (0, 220, 255), 1)
    return canvas


# ---------------------------------------------------------------
# Webcam recording (moving signs only)
# ---------------------------------------------------------------
def record_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, 30.0, "Could not open the webcam"
    start, frames = time.time(), []
    recording_from = start + 3.0                      # 3-2-1 countdown first
    while time.time() < recording_from + REC_SECONDS:
        ok, bgr = cap.read()
        if not ok:
            break
        now = time.time()
        view = bgr.copy()
        if now < recording_from:
            put(view, f"Get ready... {int(recording_from - now) + 1}", 30, 50, 1.2, (0, 255, 255), 3)
        else:
            put(view, "RECORDING - sign now!", 30, 50, 1.0, (0, 0, 255), 3)
            frames.append(bgr)
        cv2.imshow(TITLE, view)
        cv2.waitKey(1)
    cap.release()
    fps = len(frames) / REC_SECONDS if frames else 30.0
    return frames, max(fps, 5.0), ""


# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------
def main():
    import mediapipe as mp

    for f in (HAND_MODEL, CLF_MODEL, TEST_LIST):
        if not os.path.exists(f):
            raise SystemExit(f"Missing file: {f}  (see the list at the top of demo.py)")
    moving = MovingRecognizer()
    hand_kwargs = dict(base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL), num_hands=2,
                       min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3)
    video_options = mp.tasks.vision.HandLandmarkerOptions(
        running_mode=mp.tasks.vision.RunningMode.VIDEO, min_tracking_confidence=0.3, **hand_kwargs)

    paths = {p.name: p for p in Path(VIDEO_ROOT).rglob("*") if p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}}
    clips = [paths[n] for n in json.load(open(TEST_LIST)) if n in paths]
    if not clips:
        raise SystemExit("No test clips found. Check that the folder isl_emergency is next to demo.py.")
    clip_word = lambda p: p.name.split("_Crop")[0].lower()
    clips_by_word = {w: [c for c in clips if clip_word(c) == w] for w in moving.labels}
    print("Moving signs:", ", ".join(f"{i + 1}={w}" for i, w in enumerate(moving.labels)))

    # still signs are optional
    still, still_lm, pics_by_word, still_words = None, None, {}, []
    if all(os.path.exists(f) for f in (STATIC_MODEL, STATIC_LIST)) and Path(IMG_ROOT).exists():
        still = StillRecognizer()
        still_lm = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(running_mode=mp.tasks.vision.RunningMode.IMAGE, **hand_kwargs))
        for rel in json.load(open(STATIC_LIST)):
            if (Path(IMG_ROOT) / rel).exists():
                pics_by_word.setdefault(rel.split("/")[0].lower(), []).append(rel)
        still_words = [w for w in still.labels if pics_by_word.get(w)]
        print("Still signs:", ", ".join(still_words))
    else:
        print("(still-sign files not found - running with moving signs only)")

    state = {"mode": "moving", "clip": 0, "word": 0}
    words, message, msg_until, analysis, fi = [], "", 0.0, None, 0

    def splash(text):
        cv2.imshow(TITLE, render(None, 0, words, status=text, mode=state["mode"]))
        cv2.waitKey(1)

    def load_moving(path):
        splash("Analyzing the video...")
        frames, fps = read_frames(path)
        return analyze_moving(mp, video_options, moving, frames, fps, path.name, clip_word(path))

    def load_still(word):
        splash("Analyzing the pictures...")
        rels = pics_by_word[word]
        picks = random.sample(rels, min(PICTURES_PER_WORD, len(rels)))
        return analyze_still(mp, still_lm, still, [Path(IMG_ROOT) / r for r in picks],
                             f"{len(picks)} pictures", word)

    try:
        cv2.namedWindow(TITLE)
        analysis = load_moving(clips[state["clip"]])
        while True:
            msg = message if time.time() < msg_until else ""
            cv2.imshow(TITLE, render(analysis, fi, words, msg, mode=state["mode"]))
            key = cv2.waitKey(int(1000 / (analysis["fps"] if analysis else 30))) & 0xFF
            fi += 1
            if key == 255:
                continue
            ch = chr(key).lower() if 32 <= key < 127 else ""
            if key == 27 or ch == "q":
                break
            elif ch == "s":
                if still is None:
                    message, msg_until = "Still-sign files not found", time.time() + 2.5
                else:
                    state["mode"] = "still" if state["mode"] == "moving" else "moving"
                    analysis = load_still(still_words[state["word"]]) if state["mode"] == "still" \
                        else load_moving(clips[state["clip"]])
                    fi = 0
            elif state["mode"] == "moving" and ch in ("n", "p"):
                state["clip"] = (state["clip"] + (1 if ch == "n" else -1)) % len(clips)
                analysis, fi = load_moving(clips[state["clip"]]), 0
            elif state["mode"] == "moving" and ch.isdigit() and 1 <= int(ch) <= len(moving.labels):
                choices = clips_by_word.get(moving.labels[int(ch) - 1]) or clips
                analysis, fi = load_moving(random.choice(choices)), 0
            elif state["mode"] == "still" and ch in ("n", "p", "r"):
                if ch != "r":
                    state["word"] = (state["word"] + (1 if ch == "n" else -1)) % len(still_words)
                analysis, fi = load_still(still_words[state["word"]]), 0
            elif ch == " ":
                word, _, _, _ = result_of(analysis)
                if word is None:
                    message, msg_until = "Not sure about this sign - nothing added", time.time() + 2.5
                else:
                    words.append(word)
                    message, msg_until = f"Added: {word}", time.time() + 2.0
            elif key == 8 and words:
                words.pop()
            elif ch == "c":
                words.clear()
            elif ch == "w":
                if state["mode"] != "moving":
                    message, msg_until = "Webcam works in MOVING signs mode (press S)", time.time() + 3
                    continue
                frames, fps, err = record_webcam()
                if err or not frames:
                    message, msg_until = err or "No frames recorded", time.time() + 3
                else:
                    splash("Analyzing your recording...")
                    analysis, fi = analyze_moving(mp, video_options, moving, frames, fps, "webcam recording"), 0
    finally:
        if still_lm is not None:
            still_lm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()