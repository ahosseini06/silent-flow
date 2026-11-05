"""Simulate calibration on a probe dir: adapt on early segments, measure WER on the rest."""
import argparse, json, os, sys, random
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from vsr_engine import VSREngine, DecodeConfig  # noqa
from adapt import adapt  # noqa
from wer import edits, load_words, normalize, reference  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("probe"); ap.add_argument("whisper")
ap.add_argument("--calib-until", type=float, default=80.0)
ap.add_argument("--modes", default="frontend,bitfit,frontend+bitfit,encoder")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--lr", default="")
a = ap.parse_args()

words = load_words(a.whisper)
calib, test = [], []
for s in json.load(open(os.path.join(a.probe, "segments.json"))):
    ref = " ".join(normalize(reference(words, s["start"], s["end"])))
    if not ref:
        continue
    fr = np.fromfile(os.path.join(a.probe, s["file"]), dtype=np.uint8).reshape(-1, 96, 96)
    (calib if s["end"] <= a.calib_until else test).append((fr, ref))
print(f"calibration: {len(calib)} clips / {sum(len(f) for f, _ in calib)/25:.0f}s   test: {len(test)} clips")

def wer(eng):
    E = N = 0
    for fr, ref in test:
        E += edits(ref.split(), normalize(eng.transcribe(fr, DecodeConfig(beam_size=10))[0])); N += len(ref.split())
    return 100 * E / N

models = os.path.join(HERE, "..", "..", "models")
eng = VSREngine(models)
print(f"baseline test WER {wer(eng):.1f}%")
default_lr = {"frontend": 1e-4, "bitfit": 5e-4, "frontend+bitfit": 2e-4, "encoder": 2e-5}
for mode in a.modes.split(","):
    for lr in ([float(x) for x in a.lr.split(",")] if a.lr else [default_lr[mode]]):
        random.seed(0); torch.manual_seed(0)
        eng = VSREngine(models)
        val, train = calib[:2], calib[2:]
        rep = adapt(eng, train, val, mode=mode, steps=a.steps, lr=lr)
        print(f"{mode:16s} lr={lr:g}  {rep}  -> test WER {wer(eng):.1f}%")
