"""
sign_utils.py - small helpers shared by train_video_classifier.py and app.py (numpy only).

normalize_clip(clip)
    Each frame has 134 numbers = 2 hands x 67.  Per hand:  63 hand-shape numbers, wrist x/y in the picture,
    hand size, visible flag.  Close-up clips and full-body clips put the wrist at very different places and
    sizes, which would let the model guess "which dataset is this?" instead of "which sign is this?".
    This removes where the hands sit on average and how big they are over the WHOLE clip, but keeps the
    movement (how the wrist moves and how the size changes inside the clip).
    app.py must apply the SAME function to uploaded videos, so it is stored in one place.

group_split(y, groups, n_classes, rng)
    70/15/15 split per word.  Clips with the same group id (same signer, e.g. take 01 and 02) always land in
    the same part, so the test set really contains signers the model has not seen.  If every clip has its own
    group (INCLUDE has no signer info) this is an ordinary random split.
"""

import numpy as np

PER_HAND = 67


def normalize_clip(clip):
    x = np.array(clip, dtype=np.float32, copy=True)
    xy_parts, size_parts = [], []
    for k in (0, 1):
        b = k * PER_HAND
        vis = x[:, b + 66] > 0
        if vis.any():
            xy_parts.append(x[vis, b + 63:b + 65])
            size_parts.append(x[vis, b + 65])
    if not xy_parts:
        return x
    ref_xy = np.concatenate(xy_parts).mean(0)
    ref_size = max(float(np.concatenate(size_parts).mean()), 1e-4)
    for k in (0, 1):
        b = k * PER_HAND
        vis = x[:, b + 66] > 0
        if vis.any():
            x[vis, b + 63:b + 65] = np.clip((x[vis, b + 63:b + 65] - ref_xy) / ref_size, -20, 20)
            x[vis, b + 65] = np.clip(x[vis, b + 65] / ref_size, 0, 20)
    return x


def group_split(y, groups, n_classes, rng, frac=0.15):
    """Returns (train_idx, val_idx, test_idx).  Needs at least 3 groups for a word, else all go to train."""
    y, groups = np.asarray(y), np.asarray(groups)
    tr, va, te = [], [], []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        names = list(dict.fromkeys(groups[idx]))                    # unique groups, keep order
        if len(names) < 3:
            tr += list(idx)
            continue
        names = [names[i] for i in rng.permutation(len(names))]
        want = max(1, round(frac * len(idx)))
        test_n = val_n = 0
        for g in names:
            members = list(idx[groups[idx] == g])
            if test_n < want:
                te += members; test_n += len(members)
            elif val_n < want:
                va += members; val_n += len(members)
            else:
                tr += members
    return np.array(tr), np.array(va), np.array(te)