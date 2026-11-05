"""Reference alignment from Whisper word timestamps (+ re-exports of WER helpers)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from textutil import edits, normalize  # noqa: E402,F401


def load_words(whisper_json: str) -> list[tuple[float, float, str]]:
    d = json.load(open(whisper_json))
    return [(w["start"], w["end"], w["word"]) for s in d["segments"] for w in s.get("words", [])]


def reference(words, start: float, end: float) -> str:
    return " ".join(w for s, e, w in words if start <= (s + e) / 2 < end)
