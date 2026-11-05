# SilentFlow

A native macOS app that shows your webcam and writes subtitles for what your lips say.
No audio is used, and everything runs on your Mac.

```
./scripts/setup.sh       # once: Python env + model weights (~1.2 GB) + correction LLM (~2.5 GB)
./scripts/build_app.sh   # builds build/SilentFlow.app (Swift toolchain only, no Xcode)
open build/SilentFlow.app
```

Needs Apple Silicon, macOS 14+, [`uv`](https://github.com/astral-sh/uv), and the Swift toolchain (Command Line Tools is enough).

## Using it

The first launch walks you through setup: what it does, camera and Accessibility access, your push-to-talk key,
and calibration. Replay it any time from the menu bar (Show Welcome Tour…).

### As a keyboard (push-to-talk)

Hold **right Option (⌥)** in any app and mouth what you want to write. A popup at the bottom of the screen shows
the live reading. Let go, and the text lands in the popup, ready to edit. **Return** types it into the app you were
in, **Esc** throws it away, and holding **⌥** again before Return adds another phrase. The app you're typing into
stays in front the whole time. Pressing another key while holding ⌥ (a normal ⌥-shortcut) cancels the recording.

Every dictation you accept is saved with its mouth clip and the model's original reading, in
`~/Library/Application Support/SilentFlow/dictations/`. Your edits are the labels for future personal fine-tuning.

- The first launch asks for **Accessibility** access (System Settings → Privacy & Security → Accessibility). It's
  needed to see the key while other apps are focused and to type into them. The build is ad-hoc signed, so after
  a rebuild macOS treats it as a new app: remove SilentFlow from that list and allow it again.
- The menu bar icon (a mouth) lets you change the key (right ⌥/⌘/⌃/⇧), turn typing or context correction off,
  reopen the window, or calibrate. Closing the window keeps the keyboard running; Quit is in that menu.
- The camera stays on while the app runs so there's no start-up delay when you press the key.

### In the window

Face the camera and say (or silently mouth) a sentence, then pause. Subtitles stream in while you talk
(italic = still reading) and settle about a second after you stop. About half a second later a context
correction may adjust a word or two.

**Calibrate first (K).** You mouth 12 short phrases while holding Space (~1 minute). From these the app
1. sets the lip-motion sensitivity from your resting vs. mouthing lips, and
2. fine-tunes the model's visual frontend on 9 of the phrases (~30 s). It keeps the result only if it makes
   fewer mistakes on the other 3, and shows you the before/after count.

Recordings are kept in `~/Library/Application Support/SilentFlow/calibration/`, and each calibration trains on
every session so far, so calibrating again keeps improving it. Delete that folder to reset.

| Key | |
|---|---|
| hold **Space** | read while held (in the window; shows subtitles, doesn't type) |
| **K** | calibrate to your lips (R redoes the last phrase, Esc cancels) |
| **L** | context correction on/off |
| **D** | debug panel: what the model sees (the mouth crop), lip-motion meter, latency |
| **[** / **]** | less / more sensitive lip-motion detection |
| **C** | clear subtitles (and the conversation context) |

Good light on your face and clear lip movement matter more than anything else.
The model was trained on TED talks (LRS3), so it does best on natural English sentences. Single words are hard.

## How it works

```
Camera (AVFoundation, 720p)
  → face landmarks (Apple Vision)                         Swift, ~4.5 ms/frame
  → align face to a reference face, cut a 96×96 mouth crop (port of Auto-AVSR preprocessing)
  → lip-motion detector splits the stream into utterances, resampled to 25 fps
  → sidecar over a pipe: inference/server.py              PyTorch on the Mac's GPU (MPS)
       Auto-AVSR visual speech recognition (ResNet-18 + Conformer, LRS3 ~19% WER)
       joint CTC/attention beam search + transformer subword LM
       context correction: Qwen3.5-4B (MLX, 4-bit) rewrites look-alike words using the n-best + previous subtitles
  → subtitles (SwiftUI)
```

### Measured on the dev set

The dev set is a 3-minute public-domain talking-head video (437 words), with a Whisper transcript as the reference.
`inference/tools/` has the scripts.

| | WER |
|---|---|
| baseline (Auto-AVSR + its subword LM, beam 10) | 14.0% |
| + n-best rescoring with an LLM | 13.5% at best (the right word is rarely in the 10-best: oracle 11.4%) |
| + generative context correction, Qwen3.5-4B (shipped) | 13.3% |
| + same with Qwen3.5-9B | 15.1% (worse and slower) |
| calibration on the first minute, frontend fine-tune, test on the rest | 10.7% → 10.0% |
| crop zoom 0.9 / 1.1, shift ±6 px | 13.5–14.9% (noise; not calibrated) |

This video is clean, spoken-aloud, TED-like speech, which is close to the model's training data. Silent mouthing
on a webcam is further from it, and that is where calibration should matter most.

Speed on an M5 Pro: a 3-second phrase is transcribed in about 100–150 ms and a 9-second sentence in about 420 ms.
Most of that is beam search. ESPnet's stock beam search took 2–5 s for the same clips. The rewrite gives identical output and
gets its speed from:
- `inference/fast_beam.py`: a beam search with a single GPU→CPU sync per token and an exact early stop. CTC prefix scoring is vectorized in NumPy.
- `inference/fast_step.py`: real KV caches for the decoder and LM. Cross-attention keys and values are computed once per utterance and shared across the beam.

### Layout

- `Sources/SilentFlow/`: the app (`Camera`, `FaceTracker`, `MouthCropper`, `Segmenter`, `InferenceClient`, `AppModel`, UI).
- `inference/`: the sidecar. `espnet/` is a trimmed copy from [chaplin](https://github.com/amanvirparhar/chaplin) / [auto_avsr](https://github.com/mpc001/auto_avsr), patched for Python 3.12 and MPS.
- `models/`: weights, downloaded by `setup.sh` (git-ignored).

### Checking the pipeline on a video

`build/SilentFlow.app/Contents/MacOS/SilentFlow --probe <video> <out-dir>` runs the same face tracking,
cropping and segmentation code on a video file. It writes the mouth crops and one `segment_N.raw` per utterance,
which you can feed to `inference/vsr_engine.py`'s `VSREngine.transcribe`.

## Credits and licenses

SilentFlow is MIT licensed (see [LICENSE](LICENSE)). The app icon is the SilentFlow mark (`Resources/logo.svg`);
`swift scripts/make_icon.swift` regenerates `Resources/AppIcon.icns` from it.

- Model: Auto-AVSR (Ma et al., ICASSP 2023), `LRS3_V_WER19.1` and `lm_en_subword`, from
  [Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages).
  The code is Apache 2.0. The weights are trained on LRS3 and meant for research and non-commercial use; they are
  downloaded by `setup.sh`, not shipped here.
- `inference/espnet/` is adapted from ESPnet and Auto-AVSR (Apache 2.0) via chaplin (MIT), and keeps its original
  headers.
- Context correction uses Qwen3.5-4B (MLX 4-bit), downloaded by `setup.sh`.
