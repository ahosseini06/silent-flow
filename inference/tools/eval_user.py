"""Learning curve on a user's own calibration recordings.

Held-out set: each session's validation phrases. Train on increasing numbers of
the remaining phrases, for several sets of tunable parameters; report WER.
"""
import argparse, glob, json, os, random, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from vsr_engine import VSREngine, DecodeConfig  # noqa
from adapt import adapt  # noqa
from textutil import edits, normalize  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--profile", default=os.path.expanduser("~/Library/Application Support/LipReader"))
ap.add_argument("--modes", default="frontend")
ap.add_argument("--sizes", default="0,6,12,18")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--seeds", type=int, default=2)
a = ap.parse_args()

train_pool, test = [], []
for d in sorted(glob.glob(os.path.join(a.profile, "calibration", "*"))):
    meta = json.load(open(os.path.join(d, "phrases.json")))
    for i, t in enumerate(meta["texts"]):
        clip = (np.load(os.path.join(d, f"{i:02d}.npy")), t)
        (test if i in meta["val"] else train_pool).append(clip)
print(f"train pool {len(train_pool)}  test {len(test)} phrases / {sum(len(t.split()) for _, t in test)} words")

models = os.path.join(HERE, "..", "..", "models")
eng = VSREngine(models)

def wer(show=False):
    E = N = 0
    for fr, ref in test:
        hyp = eng.transcribe(fr, DecodeConfig(beam_size=10))[0]
        E += edits(ref.split(), normalize(hyp)); N += len(ref.split())
        if show: print(f"   REF {ref}\n   HYP {hyp.lower()}")
    return 100 * E / N

base = {n: p.detach().clone() for n, p in eng.model.named_parameters()}
def reset():
    with torch.no_grad():
        for n, p in eng.model.named_parameters(): p.copy_(base[n])

print(f"no adaptation: {wer(show=True):.1f}%")
lrs = {"frontend": 1e-4, "frontend+bitfit": 2e-4, "encoder": 2e-5, "bitfit": 5e-4}
for mode in a.modes.split(","):
    for size in [int(s) for s in a.sizes.split(",") if int(s) > 0]:
        res = []
        for seed in range(a.seeds):
            reset(); random.seed(seed); torch.manual_seed(seed)
            pool = train_pool[:]; random.Random(seed).shuffle(pool)
            steps = max(a.steps, 5 * size)
            adapt(eng, pool[:size], [], mode=mode, steps=steps, lr=lrs[mode])
            res.append(wer())
        print(f"{mode:16s} train={size:2d} steps={max(a.steps, 5*size)}  WER " + " / ".join(f"{r:.1f}%" for r in res) + f"  mean {np.mean(res):.1f}%")
