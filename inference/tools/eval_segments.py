"""Evaluate the lip reader on probe output: WER vs. Whisper word-timestamp references.

usage: python tools/eval_segments.py <probe-dir> <whisper.json> [--beam 10] [--no-lm]
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from vsr_engine import VSREngine, DecodeConfig  # noqa
from wer import edits, load_words, normalize, reference  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("probe"); ap.add_argument("whisper")
ap.add_argument("--beam", type=int, default=10)
ap.add_argument("--no-lm", action="store_true")
ap.add_argument("-v", action="store_true")
a = ap.parse_args()

eng = VSREngine(os.path.join(HERE, "..", "..", "models"), use_lm=not a.no_lm)
words = load_words(a.whisper)
E = N = 0; ms = []
for s in json.load(open(os.path.join(a.probe, "segments.json"))):
    fr = np.fromfile(os.path.join(a.probe, s["file"]), dtype=np.uint8).reshape(-1, 96, 96)
    ref = normalize(reference(words, s["start"], s["end"]))
    if not ref:
        continue
    text, t = eng.transcribe(fr, DecodeConfig(beam_size=a.beam))
    hyp = normalize(text)
    e = edits(ref, hyp); E += e; N += len(ref); ms.append(t["encode_ms"] + t["search_ms"])
    if a.v:
        print(f"[{s['start']:6.1f}-{s['end']:6.1f}] {e:2d}/{len(ref):2d}  REF: {' '.join(ref)}\n{'':21}HYP: {' '.join(hyp)}")
print(f"WER {100*E/N:.1f}%  ({E}/{N} words, median {np.median(ms):.0f} ms)")
