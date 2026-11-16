"""
sentence_builder.py - turns recognised sign words into an English sentence (works offline).

All rules live in sentences.json, so adding words does NOT need code changes:
  categories           group words (day, feeling, greeting, ...)
  category_templates   one sentence pattern per category      "I am {w}."
  word_overrides       a special sentence for one exact word
  combos               patterns for word sequences, longest first; "@day" means "any word of category day"

Order of rules for every position in the word list:
  1. combos (longest pattern first)  2. word_overrides  3. category template  4. just the capitalised word
"""

import json
import os

RULES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentences.json")


def _load_rules(path=RULES_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}          # missing / broken file -> plain fallback, the app keeps working


def _nice(word):
    return word.replace("_", " ")


def _show(word, cat, rules):
    """Word as shown inside a sentence; categories listed in 'capitalise_categories' (e.g. day) get a capital."""
    text = _nice(word)
    return text.title() if cat in rules.get("capitalise_categories", []) else text


def _finish(text):
    """Capitalise the first letter."""
    return text[:1].upper() + text[1:] if text else text


def _match(pattern, words, i, cats, rules):
    """Does `pattern` match words starting at i?  Returns {category: word} or None."""
    if i + len(pattern) > len(words):
        return None
    caught = {}
    for p, w in zip(pattern, words[i:i + len(pattern)]):
        if p.startswith("@"):
            cat = p[1:]
            if w not in cats.get(cat, []):
                return None
            caught[cat] = _show(w, cat, rules)
        elif p != w:
            return None
    return caught


def build_sentence(words, rules=None):
    rules = rules if rules is not None else _load_rules()
    cats = rules.get("categories", {})
    cat_tpl = rules.get("category_templates", {})
    overrides = rules.get("word_overrides", {})
    combos = sorted(rules.get("combos", []), key=lambda c: -len(c["pattern"]))

    words = [w for i, w in enumerate(words) if i == 0 or w != words[i - 1]]    # drop immediate repeats
    out, i = [], 0
    while i < len(words):
        done = False
        for c in combos:                                                        # 1. sequence rules
            caught = _match(c["pattern"], words, i, cats, rules)
            if caught is not None:
                out.append(_finish(c["text"].format(**caught)))
                i += len(c["pattern"])
                done = True
                break
        if done:
            continue
        w = words[i]
        if w in overrides:                                                      # 2. exact-word sentence
            out.append(_finish(overrides[w]))
        else:                                                                   # 3. category, 4. fallback
            cat = next((k for k, members in cats.items() if w in members), None)
            if cat in cat_tpl:
                out.append(_finish(cat_tpl[cat].format(w=_show(w, cat, rules))))
            else:
                out.append(_finish(_nice(w)) + ".")
        i += 1
    return " ".join(out)


if __name__ == "__main__":                       # quick self-test: python sentence_builder.py
    for test in (["today", "monday"], ["today", "happy"], ["loud"], ["quiet", "quiet", "loud"],
                 ["call", "doctor"], ["friday"], ["unknownword"], []):
        print(f"{test!s:32} -> {build_sentence(test)!r}")