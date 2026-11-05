"""Auto-AVSR visual speech recognition engine.

Wraps the LRS3 video-only Auto-AVSR model (Ma et al., 2023) plus its subword
transformer LM, decoded with ESPnet's batched beam search. Input is a sequence
of 96x96 grayscale mouth crops at 25 fps, already aligned to the reference face
(the native app does the face alignment and cropping).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from espnet.asr.asr_utils import get_model_conf, torch_load
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from fast_beam import beam_search

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN, STD = 0.421, 0.165
CROP = 88


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class DecodeConfig:
    beam_size: int = 8
    ctc_weight: float = 0.1
    lm_weight: float = 0.3
    nbest: int = 1


@dataclass
class Hypothesis:
    score: float  # joint CTC/attention/LM log-score from the beam search
    text: str     # UPPERCASE words, as the model emits them


class VSREngine:
    def __init__(self, models_dir: str, device: str | None = None, use_lm: bool = True):
        self.device = pick_device(device)
        vsr_dir = os.path.join(models_dir, "vsr")
        lm_dir = os.path.join(models_dir, "lm")

        with open(os.path.join(vsr_dir, "model.json"), "rb") as f:
            confs = json.load(f)
        args = argparse.Namespace(**(confs if isinstance(confs, dict) else confs[2]))

        units = os.path.join(HERE, "unigram5000_units.txt")
        with open(units) as f:
            self.token_list = ["<blank>"] + [l.split()[0] for l in f.read().splitlines()] + ["<eos>"]
        odim = len(self.token_list)

        self.model = E2E(odim, args)
        state = torch.load(os.path.join(vsr_dir, "model.pth"), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.lm = None
        if use_lm and os.path.exists(os.path.join(lm_dir, "model.pth")):
            lm_path = os.path.join(lm_dir, "model.pth")
            lm_args = get_model_conf(lm_path, os.path.join(lm_dir, "model.json"))
            lm_class = dynamic_import_lm(getattr(lm_args, "model_module", "default"), lm_args.backend)
            self.lm = lm_class(odim, lm_args)
            torch_load(lm_path, self.lm)
            self.lm.to(self.device).eval()

    @staticmethod
    def preprocess(frames: np.ndarray) -> torch.Tensor:
        """uint8 (T, 96, 96) -> float (1, T, 88, 88) normalized (C, T, H, W)."""
        t, h, w = frames.shape
        y0, x0 = (h - CROP) // 2, (w - CROP) // 2
        x = torch.from_numpy(frames[:, y0 : y0 + CROP, x0 : x0 + CROP].copy()).float().div_(255.0)
        x = (x - MEAN) / STD
        return x.unsqueeze(0)

    def ids_to_text(self, ids) -> str:
        text = "".join(self.token_list[i] for i in ids if 0 < i < len(self.token_list) - 1)
        return text.replace("\u2581", " ").strip()

    def text_to_ids(self, text: str) -> list[int]:
        """Greedy longest-match tokenization into the model's unigram units."""
        index = {t: i for i, t in enumerate(self.token_list)}
        ids = []
        for word in text.upper().split():
            piece = "\u2581" + word
            while piece:
                for end in range(len(piece), 0, -1):
                    if piece[:end] in index:
                        ids.append(index[piece[:end]])
                        piece = piece[end:]
                        break
                else:
                    piece = piece[1:]  # unknown character
        return ids

    @torch.inference_mode()
    def decode(self, frames: np.ndarray, cfg: DecodeConfig) -> tuple[list[Hypothesis], dict]:
        t0 = time.perf_counter()
        x = self.preprocess(frames).to(self.device)
        enc = self.model.encode(x)
        if self.device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        nbest = beam_search(self.model, self.lm, enc, beam=max(cfg.beam_size, cfg.nbest),
                            ctc_weight=cfg.ctc_weight, lm_weight=cfg.lm_weight, nbest=cfg.nbest)
        t2 = time.perf_counter()
        hyps = [Hypothesis(score, self.ids_to_text(ids)) for score, ids in nbest]
        return hyps, {"encode_ms": round((t1 - t0) * 1e3, 1), "search_ms": round((t2 - t1) * 1e3, 1)}

    def transcribe(self, frames: np.ndarray, cfg: DecodeConfig) -> tuple[str, dict]:
        hyps, timing = self.decode(frames, cfg)
        return (hyps[0].text if hyps else ""), timing


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING)
    eng = VSREngine(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "models"))
    rng = np.random.default_rng(0)
    for n in (40, 75, 150):
        clip = rng.integers(0, 255, size=(n, 96, 96), dtype=np.uint8)
        for beam in (4, 10):
            eng.transcribe(clip, DecodeConfig(beam_size=beam))
            text, timing = eng.transcribe(clip, DecodeConfig(beam_size=beam))
            print(f"T={n} beam={beam} {timing} -> {text!r}")
