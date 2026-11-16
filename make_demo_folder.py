"""
make_demo_folder.py - copies the held-out TEST videos (never used for training) into one folder called demo_clips,
renamed by word, so that during the demo you just pick  hello_1.MOV, today_1.MOV ...  in the upload window.

Run (from D:\\cv_new, after training):
    python make_demo_folder.py                          -> 1 test clip for every word
    set PER_WORD=2 && python make_demo_folder.py        -> up to 2 clips per word
    set WORDS=hello,how_are_you,today,monday,call,doctor && python make_demo_folder.py     -> only these words

The folder demo_clips is deleted and re-created every time, so do not keep your own files in it.
Run it again after every re-training (the test clips change when you re-train).
"""

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

INCLUDE_ROOT = os.environ.get("INCLUDE_ROOT", "include")
EMERGENCY_ROOT = os.environ.get("EMERGENCY_ROOT", "isl_emergency")
OUT = "demo_clips"
PER_WORD = int(os.environ.get("PER_WORD", 1))
WORDS = [w.strip().lower() for w in os.environ.get("WORDS", "").split(",") if w.strip()]


def include_label(rel):
    folder = Path(rel).parent
    while folder.name.strip().lower() in {"extra", "extras"} and folder.parent != folder:
        folder = folder.parent
    name = re.sub(r"^\s*\d+\s*[.\-_)]*\s*", "", folder.name)
    return re.sub(r"\s+", "_", name.strip().lower())


def main():
    if not os.path.exists("isl_test_clips.json"):
        raise SystemExit("isl_test_clips.json not found. Run train_video_classifier.py first (in this folder).")
    entries = json.load(open("isl_test_clips.json", encoding="utf-8"))

    emergency_files = {}                                           # file name -> full path
    if os.path.isdir(EMERGENCY_ROOT):
        for p in Path(EMERGENCY_ROOT).rglob("*"):
            if p.is_file():
                emergency_files[p.name] = p

    found = defaultdict(list)
    missing = 0
    for e in entries:
        e = e.replace("\\", "/")
        if "/" in e:                                               # INCLUDE: path inside the include folder
            src, word = Path(INCLUDE_ROOT) / e, include_label(e)
        else:                                                      # emergency: only a file name
            src, word = emergency_files.get(e), e.split("_Crop")[0].lower()
        if src is None or not Path(src).exists():
            missing += 1
            continue
        found[word].append(Path(src))

    if WORDS:
        unknown = [w for w in WORDS if w not in found]
        if unknown:
            print("No test clip found for:", ", ".join(unknown), "(check spelling)")
        found = {w: found[w] for w in WORDS if w in found}

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    n = 0
    for word in sorted(found):
        for k, src in enumerate(found[word][:PER_WORD], 1):
            shutil.copy2(src, Path(OUT) / f"{word}_{k}{src.suffix.lower()}")
            n += 1
    print(f"Copied {n} test clips for {len(found)} words into '{OUT}'.")
    if missing:
        print(f"{missing} entries in isl_test_clips.json could not be found on disk "
              f"(are the '{INCLUDE_ROOT}' and '{EMERGENCY_ROOT}' folders in this directory?)")
    print("In the app's upload window, open the demo_clips folder and pick the files. Tip: type  hello*  in the "
          "File name box and press Enter to filter.")


if __name__ == "__main__":
    main()