"""Lip-reading inference sidecar for the native app.

Protocol (stdin/stdout). Every request is one JSON header line, optionally
followed by raw uint8 96x96 grayscale mouth crops (25 fps):

  {"type": "decode", "id": int, "frames": N, "beam": int, "final": bool}   + N frames
      -> {"type": "result", "id", "text", "final", "corrected": false, "done": bool, ...timing}
         for finals with the LLM loaded, a second message follows once the
         context correction ran: {"type": "result", ..., "corrected": true, "done": true}
  {"type": "calibrate", "phrases": [{"text": str, "frames": n}, ...], "val": [idx, ...]}
      + all phrases' frames back to back
      -> {"type": "progress", "step", "total"} ... then {"type": "calibrated", ...report}
  {"type": "label", "ids": [segment ids], "model_text": str, "final_text": str}
      the user accepted a dictation (possibly after editing it): store the clip
      with its final text under <profile>/dictations/ for later fine-tuning
  {"type": "reset_context"}             forget previous subtitles (LLM context)
  {"type": "config", "llm": bool}       turn context correction on/off

Startup: {"type": "ready", "device", "profile": bool}, later {"type": "llm_ready"}.
Logs go to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from adapt import DEFAULT_MODE, adapt, adapted_state, load_adapted_state  # noqa: E402
from textutil import edits, normalize, tidy  # noqa: E402
from vsr_engine import DecodeConfig, VSREngine  # noqa: E402

SIDE = 96
FRAME = SIDE * SIDE
CONTEXT_TTL = 120.0  # seconds of silence after which the conversation context resets


def log(msg: str) -> None:
    print(f"[lipread] {msg}", file=sys.stderr, flush=True)


_send_lock = threading.Lock()


def send(obj: dict) -> None:
    with _send_lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def read_exact(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class Server:
    def __init__(self, args):
        self.engine = VSREngine(args.models, device=args.device, use_lm=not args.no_lm)
        self.profile_dir = os.path.expanduser(args.profile)
        os.makedirs(self.profile_dir, exist_ok=True)
        self.base_state = adapted_state(self.engine, DEFAULT_MODE)  # pristine copy of tunable weights
        self.has_profile = self._load_profile()

        self.llm = None
        self.llm_enabled = not args.no_llm
        self.context: list[str] = []
        self.recent: dict[int, np.ndarray] = {}  # last final clips, for labelling
        self.last_final = 0.0
        # MLX streams are per-thread: load *and* run the LLM on one dedicated thread.
        self.llm_thread = ThreadPoolExecutor(max_workers=1)
        if not args.no_llm:
            self.llm_thread.submit(self._load_llm, args.llm_repo)

    # ---------- setup

    def _profile_path(self) -> str:
        return os.path.join(self.profile_dir, "adapted.pt")

    def _load_profile(self) -> bool:
        path = self._profile_path()
        if not os.path.exists(path):
            return False
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
            load_adapted_state(self.engine, blob["state"])
            log(f"loaded personal calibration ({blob.get('meta', {})})")
            return True
        except Exception as e:  # a broken profile must never stop the app
            log(f"ignoring unreadable profile: {e!r}")
            return False

    def _load_llm(self, repo: str) -> None:
        try:
            from llm_correct import LLMCorrector

            t = time.perf_counter()
            self.llm = LLMCorrector(repo)
            log(f"context correction ready ({repo}) in {time.perf_counter() - t:.1f}s")
            send({"type": "llm_ready"})
        except Exception as e:
            log(f"context correction unavailable: {e!r}")

    def warm_up(self) -> None:
        warm = np.random.default_rng(0).integers(0, 255, size=(50, SIDE, SIDE), dtype=np.uint8)
        for beam in (4, 10):
            self.engine.decode(warm, DecodeConfig(beam_size=beam, nbest=5))

    # ---------- requests

    def decode(self, req: dict, frames: np.ndarray) -> None:
        rid, final = req.get("id"), bool(req.get("final", True))
        quiet = bool(req.get("quiet", False))  # calibration preview: no LLM, no conversation context
        if final:
            self.recent[rid] = frames
            while len(self.recent) > 16:
                self.recent.pop(next(iter(self.recent)))
        use_llm = final and not quiet and self.llm is not None and self.llm_enabled
        if len(frames) < 10:
            send({"type": "result", "id": rid, "text": "", "final": final, "corrected": False, "done": True})
            return
        cfg = DecodeConfig(beam_size=int(req.get("beam", 10 if final else 4)), nbest=5 if use_llm else 1)
        try:
            hyps, timing = self.engine.decode(frames, cfg)
        except Exception as e:  # keep the sidecar alive no matter what
            log(f"inference error: {e!r}")
            hyps, timing = [], {"encode_ms": 0, "search_ms": 0}
        top = hyps[0].text if hyps else ""
        log(f"#{rid} {'final' if final else 'partial'} T={len(frames)} {timing} -> {top!r}")
        use_llm = use_llm and bool(top)
        send({"type": "result", "id": rid, "text": tidy(top), "final": final, "corrected": False,
              "done": not use_llm, **timing})
        if not final or quiet:
            return

        now = time.monotonic()
        if now - self.last_final > CONTEXT_TTL:
            self.context.clear()
        self.last_final = now
        text = top.lower()
        if use_llm:
            try:
                fixed, t = self.llm_thread.submit(
                    self.llm.correct, [h.text.lower() for h in hyps], list(self.context)).result()
            except Exception as e:
                fixed, t = None, {"error": repr(e)}
            if fixed:
                log(f"#{rid} corrected -> {fixed!r} {t}")
                text = fixed
            elif "rejected" in t:
                log(f"#{rid} rejected rewrite {t['rejected']!r}")
            send({"type": "result", "id": rid, "text": tidy(text), "final": True, "corrected": True,
                  "done": True, **t})
        if text:
            self.context = (self.context + [text])[-6:]

    def label(self, req: dict) -> None:
        clips = [self.recent[i] for i in req.get("ids", []) if i in self.recent]
        final_text = " ".join(normalize(req.get("final_text", "")))
        if not clips or not final_text:
            return
        d = os.path.join(self.profile_dir, "dictations")
        os.makedirs(d, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + f"-{req['ids'][0]}"
        np.save(os.path.join(d, name + ".npy"), np.concatenate(clips))
        model_text = " ".join(normalize(req.get("model_text", "")))
        with open(os.path.join(d, name + ".json"), "w") as f:
            json.dump({"model_text": model_text, "final_text": final_text,
                       "edited": model_text != final_text, "time": time.time()}, f)
        log(f"saved dictation {name} ({'edited' if model_text != final_text else 'accepted as is'})")

    def calibrate(self, req: dict, payload: bytes) -> None:
        phrases = req.get("phrases", [])
        clips, off = [], 0
        for p in phrases:
            n = int(p["frames"])
            clips.append((np.frombuffer(payload[off : off + n * FRAME], dtype=np.uint8).reshape(n, SIDE, SIDE),
                          " ".join(normalize(p["text"]))))
            off += n * FRAME
        val_idx = set(int(i) for i in req.get("val", []))
        session = self._save_session(clips, sorted(val_idx))

        # Train on every phrase ever recorded except this session's held-out ones.
        train = [c for i, c in enumerate(clips) if i not in val_idx] + self._older_sessions(exclude=session)
        val = [c for i, c in enumerate(clips) if i in val_idx]
        log(f"calibrating on {len(train)} phrases, validating on {len(val)}")

        load_adapted_state(self.engine, self.base_state)  # always adapt from the original model
        wer_before = self._wer(val)
        report = adapt(self.engine, train, val, mode=DEFAULT_MODE, steps=60, lr=1e-4,
                       on_step=lambda s, total, _l: send({"type": "progress", "step": s, "total": total}))
        wer_after = self._wer(val) if report["kept"] else wer_before
        # Held-out loss decides; with only ~25 test words one extra mistake is noise,
        # but clearly more mistakes means don't trust it.
        if report["kept"] and wer_after[0] > wer_before[0] + max(1, round(0.05 * wer_before[1])):
            report["kept"] = False
            load_adapted_state(self.engine, self.base_state)

        if report["kept"]:
            torch.save({"state": adapted_state(self.engine, DEFAULT_MODE),
                        "meta": {"phrases": len(train), "time": time.strftime("%Y-%m-%d %H:%M")}},
                       self._profile_path())
            self.has_profile = True
        elif self.has_profile:
            self._load_profile()  # keep using the previous calibration
        log(f"calibration {report}; val errors {wer_before} -> {wer_after}")
        send({"type": "calibrated", **report, "errors_before": wer_before[0], "errors_after": wer_after[0],
              "words": wer_before[1], "train_phrases": len(train)})

    def _wer(self, clips) -> tuple[int, int]:
        e = n = 0
        for frames, text in clips:
            hyp, _ = self.engine.transcribe(frames, DecodeConfig(beam_size=10))
            e += edits(text.split(), normalize(hyp))
            n += len(text.split())
        return e, n

    def _sessions_dir(self) -> str:
        d = os.path.join(self.profile_dir, "calibration")
        os.makedirs(d, exist_ok=True)
        return d

    def _save_session(self, clips, val_idx) -> str:
        name = time.strftime("%Y%m%d-%H%M%S")
        d = os.path.join(self._sessions_dir(), name)
        os.makedirs(d, exist_ok=True)
        for i, (frames, _) in enumerate(clips):
            np.save(os.path.join(d, f"{i:02d}.npy"), frames)
        with open(os.path.join(d, "phrases.json"), "w") as f:
            json.dump({"texts": [t for _, t in clips], "val": val_idx}, f, indent=1)
        return name

    def _older_sessions(self, exclude: str) -> list:
        out = []
        root = self._sessions_dir()
        for name in sorted(os.listdir(root)):
            meta = os.path.join(root, name, "phrases.json")
            if name == exclude or not os.path.exists(meta):
                continue
            texts = json.load(open(meta))["texts"]
            for i, t in enumerate(texts):  # older held-out phrases are fair game for training now
                path = os.path.join(root, name, f"{i:02d}.npy")
                if os.path.exists(path):
                    out.append((np.load(path), t))
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(HERE, "..", "models"))
    ap.add_argument("--profile", default="~/Library/Application Support/LipReader")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="disable LLM context correction")
    ap.add_argument("--llm-repo", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    warnings.filterwarnings("ignore")

    t = time.perf_counter()
    server = Server(args)
    server.warm_up()
    log(f"model ready on {server.engine.device} in {time.perf_counter() - t:.1f}s"
        f"{' (personal calibration loaded)' if server.has_profile else ''}")
    send({"type": "ready", "device": str(server.engine.device), "profile": server.has_profile})

    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            log(f"bad header: {line[:200]!r}")
            continue
        kind = req.get("type", "decode")
        if kind == "decode":
            n = int(req.get("frames", 0))
            payload = read_exact(stdin, n * FRAME) if n > 0 else b""
            if payload is None:
                break
            server.decode(req, np.frombuffer(payload, dtype=np.uint8).reshape(n, SIDE, SIDE))
        elif kind == "calibrate":
            total = sum(int(p["frames"]) for p in req.get("phrases", []))
            payload = read_exact(stdin, total * FRAME)
            if payload is None:
                break
            try:
                server.calibrate(req, payload)
            except Exception as e:
                log(f"calibration failed: {e!r}")
                send({"type": "calibrated", "kept": False, "error": str(e)})
        elif kind == "label":
            try:
                server.label(req)
            except Exception as e:
                log(f"couldn't save dictation: {e!r}")
        elif kind == "reset_context":
            server.context.clear()
        elif kind == "config":
            if "llm" in req:
                server.llm_enabled = bool(req["llm"])
                log(f"context correction {'on' if server.llm_enabled else 'off'}")


if __name__ == "__main__":
    main()
