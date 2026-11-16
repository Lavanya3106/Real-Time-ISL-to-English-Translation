"""
app.py - ISL Sign Language Translator (video upload GUI)

Upload one or more sign-language video clips -> the model recognises each sign
-> the signs are joined into an English sentence.

Files needed in the SAME folder:
  app.py, sentence_builder.py, sentences.json, extract_hand_landmarks.py, hand_landmarker.task, isl_model.pt

Install:  pip install streamlit mediapipe opencv-python numpy torch
Run:      streamlit run app.py
"""

import os
import tempfile

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from extract_hand_landmarks import PER_HAND, result_to_frame   # same folder
from sentence_builder import build_sentence                    # same folder (+ sentences.json)
from sign_utils import normalize_clip                          # same folder

HAND_MODEL = "hand_landmarker.task"
CLF_MODEL = "isl_model.pt"
CONF_MIN = 0.50          # below this the app says "not sure"
MAX_SIDE = 1280           # big videos are shrunk to this for speed
MAX_PREVIEW = 200        # max frames kept for the skeleton preview

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

# ---------- word -> sentence (rules live in sentences.json, code in sentence_builder.py) ----------


# ---------- model (same design as the training script) ----------
class SignGRU(nn.Module):
    def __init__(self, n_classes, d_in, hidden=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(hidden * 4, n_classes))

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.head(torch.cat([h.mean(1), h.max(1).values], dim=1))


def resample(x, n):
    t = len(x)
    pos = np.linspace(0, t - 1, n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (pos - lo)[:, None]
    return (x[lo] * (1 - w) + x[hi] * w).astype(np.float32)


@st.cache_resource
def load_model():
    ck = torch.load(CLF_MODEL, map_location="cpu")
    model = SignGRU(len(ck["labels"]), ck["d_in"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["labels"], ck["fixed_len"], ck.get("norm", False)


def make_options():
    import mediapipe as mp
    return mp, mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3)


def make_gif(frames, fps):
    """Small animated GIF of the clip. It plays in every browser, so it is used when the video file type
    (for example .avi) cannot be played by the browser itself."""
    if not frames:
        return None
    import io
    from PIL import Image
    imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], loop=0, duration=max(20, int(2000 / fps)))
    return buf.getvalue()


@torch.no_grad()
def analyze(video_bytes, suffix):
    """Video bytes -> dict(probs, seen, preview frames, fps)."""
    model, labels, fixed_len, norm = load_model()
    mp, options = make_options()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        path = tmp.name
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps == fps and fps >= 1 else 30.0
    feats, preview, thumbs, i = [], [], [], 0
    step = 1
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as lm:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            if max(h, w) > MAX_SIDE:
                s = MAX_SIDE / max(h, w)
                bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
                h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if i % 2 == 0 and len(thumbs) < 150:             # small copy (before drawing) for the animated preview
                thumbs.append(cv2.resize(rgb, (480, max(1, int(480 * h / w))), interpolation=cv2.INTER_AREA)
                              if w > 480 else rgb.copy())
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
            feats.append(result_to_frame(res))
            if i % step == 0 and len(preview) < MAX_PREVIEW:
                for k, hand in enumerate(res.hand_landmarks):
                    pts = [(int(p.x * w), int(p.y * h)) for p in hand[:21]]
                    col = (80, 220, 80) if k == 0 else (255, 160, 60)
                    for a, b in BONES:
                        cv2.line(rgb, pts[a], pts[b], col, 2, cv2.LINE_AA)
                    for p in pts:
                        cv2.circle(rgb, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
                preview.append(cv2.resize(rgb, (640, max(1, int(640 * h / w))), interpolation=cv2.INTER_AREA)
                               if w > 640 else rgb)             # smaller frames use less memory
            i += 1
    cap.release()
    os.remove(path)
    gif = make_gif(thumbs, fps)

    feats = np.array(feats, dtype=np.float32)
    if len(feats) < 5:
        return {"probs": None, "seen": 0.0, "preview": preview, "labels": labels, "fps": fps, "gif": gif}
    seen = float(((feats[:, PER_HAND - 1] > 0) | (feats[:, 2 * PER_HAND - 1] > 0)).mean())
    probs = None
    if seen >= 0.2:
        if norm:                                          # same step the training script used
            feats = normalize_clip(feats)
        x = torch.tensor(resample(feats, fixed_len))[None]
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    return {"probs": probs, "seen": seen, "preview": preview, "labels": labels, "fps": fps, "gif": gif}


# ---------------------------------------------------------------
# GUI
# ---------------------------------------------------------------
import html

from sentence_builder import _load_rules      # same folder (used to group the vocabulary list)

st.set_page_config(page_title="ISL Sign Translator", page_icon="🤟", layout="wide")

BROWSER_VIDEO = {".mp4", ".mov", ".m4v", ".webm"}      # file types a browser can play; others get an animated preview

SERIF = '"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif'

CSS = """
<style>
.block-container{max-width:1040px;padding-top:1.6rem;}
header[data-testid="stHeader"]{background:transparent;}
.stAppDeployButton,[data-testid="stAppDeployButton"]{display:none;}

.hero{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:1rem 2rem;
  background:#14213D;border-radius:20px;padding:2.1rem 2.4rem;margin-bottom:1.5rem;}
.hero-text{flex:1 1 320px;min-width:260px;}
.hero-title{font-family:SERIF_STACK;font-weight:600;font-size:2.35rem;line-height:1.15;color:#fff;
  letter-spacing:-.01em;margin-bottom:.7rem;}
.hero-sub{color:#C3CCE2;font-size:1.02rem;line-height:1.55;max-width:33rem;}
.hero svg{flex:0 1 340px;width:340px;max-width:100%;height:auto;}

.tr{background:#E6EDFC;border-radius:14px;padding:1.5rem 1.7rem 1.35rem;margin:.6rem 0 1.4rem;}
.tr-text{font-family:SERIF_STACK;font-size:2.15rem;line-height:1.3;color:#14213D;font-weight:600;margin-bottom:1.1rem;}
.tr-empty{color:#3F4D6B;font-size:1.02rem;line-height:1.5;margin-bottom:1rem;}
.tr-note{margin-top:.9rem;font-size:.88rem;color:#3F4D6B;}
.seq{display:flex;flex-wrap:wrap;align-items:center;row-gap:.5rem;}
.link{display:inline-block;width:1.1rem;height:2px;background:#9FB0D3;}
.sign{display:inline-flex;align-items:center;gap:.5rem;padding:.22rem .85rem .22rem .25rem;border-radius:999px;
  background:#fff;border:1px solid #C3D0EC;font-size:.92rem;color:#14213D;}
.sign .n{display:inline-flex;align-items:center;justify-content:center;width:1.5rem;height:1.5rem;border-radius:50%;
  background:#14213D;color:#fff;font-size:.78rem;font-weight:600;}
.sign.warn{border:1px dashed #E08A00;background:#FFF8EC;color:#6B4200;}
.sign.warn .n{background:#E08A00;}
.sign.bad{border:1px dashed #C8374B;background:#FFF1F3;color:#7E1B2B;}
.sign.bad .n{background:#C8374B;}

.clip-title{font-weight:600;color:#14213D;font-size:1.05rem;margin-bottom:.2rem;}
.fname{color:#66738F;font-weight:400;font-size:.88rem;margin-left:.6rem;}
.word{font-family:SERIF_STACK;font-size:2.3rem;font-weight:600;line-height:1.15;color:#14213D;margin-bottom:.45rem;}
.pill{display:inline-block;padding:.16rem .7rem;border-radius:999px;font-size:.85rem;font-weight:600;}
.pill.ok{background:#D9F3E5;color:#10603A;}
.pill.warn{background:#FFE9C4;color:#6B4200;}
.pill.bad{background:#FFDCE1;color:#7E1B2B;}
.hint{font-size:.92rem;color:#3F4D6B;margin-top:.6rem;line-height:1.5;}
.bars{margin-top:1.1rem;}
.bar{display:grid;grid-template-columns:7.5rem 1fr 2.6rem;align-items:center;gap:.6rem;font-size:.9rem;color:#14213D;margin:.34rem 0;}
.bar .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.track{height:8px;border-radius:99px;background:#E1E7F3;overflow:hidden;}
.fill{height:100%;border-radius:99px;background:#9FB0D3;}
.fill.ok{background:#1E9E62;}
.fill.warn{background:#E08A00;}
.val{text-align:right;color:#3F4D6B;font-variant-numeric:tabular-nums;}
.meta{font-size:.85rem;color:#66738F;margin-top:.9rem;}

.start{background:#fff;border:1px solid #D9E0EE;border-radius:14px;padding:1.3rem 1.6rem;margin-top:1.1rem;}
.start-title{font-family:SERIF_STACK;font-size:1.3rem;font-weight:600;color:#14213D;margin-bottom:.3rem;}
.start ul{margin:.4rem 0 0 1.1rem;padding:0;color:#2B3A5A;line-height:1.75;}
.start-eg{margin-top:1.4rem;color:#3F4D6B;font-size:.95rem;}

.vg{font-size:.82rem;font-weight:600;color:#3F4D6B;margin:.7rem 0 .3rem;}
.vocab{display:flex;flex-wrap:wrap;gap:.3rem;}
.vw{padding:.1rem .6rem;border-radius:999px;background:#EAEFFA;color:#14213D;font-size:.82rem;}
</style>
""".replace("SERIF_STACK", SERIF)


def nice(word):
    return word.replace("_", " ")


def hand_svg():
    """Two hand-tracking skeletons (same colours the app draws on your videos)."""
    base = [(100, 200), (72, 180), (52, 155), (38, 132), (28, 110), (80, 130), (76, 95), (74, 70), (72, 48),
            (100, 125), (100, 85), (100, 58), (100, 34), (120, 130), (124, 95), (126, 70), (128, 50),
            (138, 148), (148, 120), (154, 100), (158, 82)]

    def hand(dx, dy, mirror, color, angle):
        pts = [((200 - x) if mirror else x, y) for x, y in base]
        pts = [(x + dx, y + dy) for x, y in pts]
        lines = "".join(f'<line x1="{pts[a][0]}" y1="{pts[a][1]}" x2="{pts[b][0]}" y2="{pts[b][1]}"/>' for a, b in BONES)
        dots = "".join(f'<circle cx="{x}" cy="{y}" r="3.4"/>' for x, y in pts)
        cx, cy = (100 if not mirror else 100) + dx, 120 + dy
        return (f'<g transform="rotate({angle} {cx} {cy})" stroke="{color}" stroke-width="2.4" stroke-linecap="round">'
                f'{lines}<g fill="#14213D" stroke-width="2">{dots}</g></g>')

    return ('<svg viewBox="0 0 400 240" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
            + hand(0, 22, False, "#50DC78", -9) + hand(190, 0, True, "#FFA03C", 9) + '</svg>')


HERO = ('<div class="hero"><div class="hero-text">'
        '<div class="hero-title" role="heading" aria-level="1">Indian Sign Language Translator</div>'
        '<div class="hero-sub">Upload short sign videos. The app finds the hands, recognises each sign '
        'and writes the sentence in English.</div></div>' + hand_svg() + '</div>')

GROUP_NAMES = {"day": "Days", "time_ref": "Today, tomorrow, yesterday", "time_unit": "Units of time",
               "pronoun": "Pronouns", "greeting": "Greetings", "feeling": "Feelings",
               "describing": "Describing words", "person_adj": "Describing people", "family": "Family",
               "thing": "Home and places"}


def vocab_html(labels):
    def group(title, words):
        chips = "".join(f'<span class="vw">{html.escape(nice(w))}</span>' for w in words)
        return f'<div class="vg">{html.escape(title)}</div><div class="vocab">{chips}</div>'

    cats = _load_rules().get("categories", {})
    seen, parts = set(), []
    for cat, members in cats.items():
        ws = [w for w in members if w in labels]
        if ws:
            seen.update(ws)
            parts.append(group(GROUP_NAMES.get(cat, cat.replace("_", " ").capitalize()), ws))
    rest = [w for w in labels if w not in seen]
    if rest:
        parts.append(group("Emergency and other words", rest))
    return "".join(parts)


def sequence_html(items):
    """items = [(status, text)] in signing order; status is ok / warn / bad."""
    chips = []
    for i, (status, text) in enumerate(items, 1):
        cls = "" if status == "ok" else f" {status}"
        chips.append(f'<span class="sign{cls}"><span class="n">{i}</span>{html.escape(text)}</span>')
    return '<div class="seq">' + '<span class="link"></span>'.join(chips) + "</div>"


def verdict_html(c, conf_min):
    a, status = c["a"], c["status"]
    if status == "bad":
        top = ('<div class="word">No hands found</div><span class="pill bad">Not recognised</span>'
               '<div class="hint">Both hands need to stay in view and be well lit. '
               'Try recording the sign again, a little closer to the camera.</div>')
        return top
    p = c["p"]
    if status == "ok":
        top = (f'<div class="word">{html.escape(nice(c["best"]))}</div>'
               f'<span class="pill ok">{100 * p:.0f}% confident</span>')
    else:
        top = (f'<div class="word">{html.escape(nice(c["best"]))}</div>'
               f'<span class="pill warn">Not sure, {100 * p:.0f}%</span>'
               f'<div class="hint">Left out of the translation because the confidence is below '
               f'{100 * conf_min:.0f}%.</div>')
    bars = ""
    for rank, k in enumerate(c["order"][:3]):
        pk = float(a["probs"][k])
        cls = f" {status}" if rank == 0 else ""
        bars += (f'<div class="bar"><span class="nm">{html.escape(nice(a["labels"][k]))}</span>'
                 f'<div class="track"><div class="fill{cls}" style="width:{100 * pk:.0f}%"></div></div>'
                 f'<span class="val">{100 * pk:.0f}%</span></div>')
    seen = f'Hands visible in {100 * a["seen"]:.0f}% of frames.'
    if a["seen"] < 0.5:
        seen += " That is low. Better light usually helps."
    return top + f'<div class="bars">{bars}</div><div class="meta">{seen}</div>'


# ---------- page ----------
st.markdown(CSS, unsafe_allow_html=True)

for f in (HAND_MODEL, CLF_MODEL):
    if not os.path.exists(f):
        st.error(f"Missing file: {f}. Put it in the same folder as app.py.")
        st.stop()

_, LABELS, _, _ = load_model()

with st.sidebar:
    st.markdown("### Settings")
    conf_min = st.slider("Minimum confidence", 0.2, 0.9, CONF_MIN, 0.05,
                         help="Signs recognised with less confidence than this are left out of the sentence.")
    with st.expander(f"Vocabulary ({len(LABELS)} signs)"):
        st.markdown(vocab_html(LABELS), unsafe_allow_html=True)
    with st.expander("How it works"):
        st.write("MediaPipe finds both hands in every frame. A neural network (a GRU) reads the hand movement "
                 "and picks the sign. Simple rules in sentences.json turn the signs into a sentence. "
                 "Everything runs on this computer and needs no internet.")

st.markdown(HERO, unsafe_allow_html=True)

files = st.file_uploader("Upload sign videos, in the order they were signed",
                         type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True,
                         help="One sign per video. Keep both hands in view.")

if not files:
    example = ""
    if "today" in LABELS and "monday" in LABELS:
        example = ('<div class="start-eg">For example, one video of today followed by one of monday reads like this:</div>'
                   f'<div class="tr"><div class="tr-text">{html.escape(build_sentence(["today", "monday"]))}</div>'
                   + sequence_html([("ok", "today"), ("ok", "monday")]) + '</div>')
    st.markdown('<div class="start"><div class="start-title">Start with one video</div><ul>'
                '<li>Pick one sign per video, with both hands in view.</li>'
                '<li>Upload several videos in the order they were signed to build a sentence.</li>'
                '<li>Videos are processed on this computer. Nothing is sent to the internet.</li></ul></div>'
                + example, unsafe_allow_html=True)
    st.stop()

# ---- read every clip first (cached), so the translation can sit at the top ----
clips = []
for f in files:
    data = f.getvalue()
    key = f"{f.name}_{len(data)}"
    if key not in st.session_state:                       # don't re-analyze on every click
        with st.spinner(f"Reading {f.name} ..."):
            st.session_state[key] = analyze(data, os.path.splitext(f.name)[1] or ".mp4")
    a = st.session_state[key]
    if a["probs"] is None:
        c = dict(status="bad", best=None, p=0.0, order=None)
    else:
        order = np.argsort(-a["probs"])
        p = float(a["probs"][order[0]])
        c = dict(status="ok" if p >= conf_min else "warn", best=a["labels"][order[0]], p=p, order=order)
    c.update(f=f, data=data, key=key, a=a)
    clips.append(c)

words = [c["best"] for c in clips if c["status"] == "ok"]
items = [(c["status"], nice(c["best"]) if c["status"] == "ok" else ("not sure" if c["status"] == "warn" else "no hands"))
         for c in clips]

if words:
    body = f'<div class="tr-text">{html.escape(build_sentence(words))}</div>'
else:
    body = ('<div class="tr-empty">No sign was recognised with enough confidence. Lower the minimum confidence '
            'in the sidebar, or try a clearer video.</div>')
left_out = len(clips) - len(words)
note = (f'<div class="tr-note">{left_out} of {len(clips)} videos left out. The details are below.</div>'
        if words and left_out else "")
st.markdown(f'<div class="tr">{body}{sequence_html(items)}{note}</div>', unsafe_allow_html=True)

st.subheader("Videos")
for n, c in enumerate(clips, 1):
    with st.container(border=True):
        st.markdown(f'<div class="clip-title">Video {n}<span class="fname">{html.escape(c["f"].name)}</span></div>',
                    unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")
        with left:
            ext = os.path.splitext(c["f"].name)[1].lower()
            if ext in BROWSER_VIDEO or not c["a"].get("gif"):
                st.video(c["data"])
            else:
                st.image(c["a"]["gif"])
                st.caption("The browser cannot play this file type, so this is an animated preview of the video.")
        with right:
            st.markdown(verdict_html(c, conf_min), unsafe_allow_html=True)
        if c["a"]["preview"]:
            with st.expander("Hand tracking"):
                frames = c["a"]["preview"]
                idx = st.slider("Frame", 0, len(frames) - 1, 0, key=f"s_{n}_{c['key']}") if len(frames) > 1 else 0
                st.image(frames[idx])