"""Downloads the model weights SilentFlow needs into one folder.

The packaged app (scripts/package.sh) ships without weights, because together they are ~4.3 GB
and a GitHub release file can be at most 2 GB. On first launch the app runs this script with the
bundled Python and shows its progress. A developer checkout uses scripts/setup.sh instead.

    python fetch_models.py --dest ~/Library/Application\\ Support/SilentFlow/models
    python fetch_models.py --dest /tmp/m --list        # print the plan, download nothing

Writes one JSON object per line to stdout:
    {"type": "plan", "total": bytes}
    {"type": "download", "done": bytes, "total": bytes}   (about twice a second)
    {"type": "done"}
    {"type": "error", "message": "..."}                   (exit code 1)

Files are pinned to exact Hugging Face commits, downloaded to <name>.part and resumed from there
if interrupted, and checked against their expected size. A `.complete` marker is written last, so
the app only starts the engine once every file is in place.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

try:  # use the macOS keychain's certificates; the bundled Python has no CA store of its own
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

VSR = ("Amanvir/LRS3_V_WER19.1", "c0b413f56b164565865a63f00774ee55a9738cc5")
LM = ("Amanvir/lm_en_subword", "f4e39b6d0904b2a900c53a2fff0377796c797bdc")
LLM = ("mlx-community/Qwen3.5-4B-MLX-4bit", "32f3e8ecf65426fc3306969496342d504bfa13f3")
LLM_DIR = "llm/Qwen3.5-4B-MLX-4bit"

# (repo, revision, file in repo, path under --dest, size in bytes)
FILES = [
    (*VSR, "model.json", "vsr/model.json", 5512),
    (*VSR, "model.pth", "vsr/model.pth", 1001908942),
    (*LM, "model.json", "lm/model.json", 951),
    (*LM, "model.pth", "lm/model.pth", 215076928),
    *[(*LLM, f, f"{LLM_DIR}/{f}", n) for f, n in [
        ("config.json", 3366),
        ("model.safetensors", 3034300695),
        ("model.safetensors.index.json", 101944),
        ("tokenizer.json", 19989343),
        ("tokenizer_config.json", 1139),
        ("chat_template.jinja", 7756),
        ("vocab.json", 6722759),
        ("preprocessor_config.json", 390),
        ("processor_config.json", 1300),
        ("video_preprocessor_config.json", 385),
    ]],
]
MARKER = ".complete"
VERSION = "1"  # bump when FILES changes, so existing installs fetch the new set


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def complete(dest: str) -> bool:
    try:
        with open(os.path.join(dest, MARKER)) as f:
            return f.read().strip() == VERSION
    except OSError:
        return False


def have(path: str, size: int) -> int:
    """Bytes already on disk for this file (finished or partial)."""
    if os.path.exists(path) and os.path.getsize(path) == size:
        return size
    part = path + ".part"
    return min(os.path.getsize(part), size) if os.path.exists(part) else 0


def fetch(url: str, path: str, size: int, on_bytes) -> None:
    part = path + ".part"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    start = os.path.getsize(part) if os.path.exists(part) else 0
    if start > size:
        os.remove(part)
        start = 0
    req = urllib.request.Request(url, headers={"User-Agent": "SilentFlow"})
    if start:
        req.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(req, timeout=60) as r:
        if start and r.status != 206:  # server ignored the range: start over
            start = 0
        with open(part, "ab" if start else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                on_bytes(len(chunk))
    got = os.path.getsize(part)
    if got != size:
        raise IOError(f"{os.path.basename(path)}: got {got} bytes, expected {size}")
    os.replace(part, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args()
    dest = os.path.expanduser(args.dest)

    if complete(dest):
        send({"type": "done"})
        return 0

    total = sum(f[4] for f in FILES)
    done = sum(have(os.path.join(dest, f[3]), f[4]) for f in FILES)
    send({"type": "plan", "total": total, "done": done})
    if args.list:
        for repo, rev, name, rel, size in FILES:
            print(f"{size:>12}  {repo}@{rev[:7]}/{name} -> {rel}", file=sys.stderr)
        return 0

    last = 0.0
    progress = {"done": done}

    def on_bytes(n: int) -> None:
        nonlocal last
        progress["done"] += n
        now = time.monotonic()
        if now - last > 0.5:
            last = now
            send({"type": "download", "done": progress["done"], "total": total})

    try:
        for repo, rev, name, rel, size in FILES:
            path = os.path.join(dest, rel)
            if os.path.exists(path) and os.path.getsize(path) == size:
                continue
            for attempt in range(4):
                try:
                    fetch(f"https://huggingface.co/{repo}/resolve/{rev}/{name}", path, size, on_bytes)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError, IOError):
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
                finally:  # re-count from disk so a restarted or retried file can't skew the total
                    progress["done"] = sum(have(os.path.join(dest, f[3]), f[4]) for f in FILES)
        with open(os.path.join(dest, MARKER), "w") as f:
            f.write(VERSION)
    except Exception as e:  # noqa: BLE001 - every failure is reported to the app the same way
        send({"type": "error", "message": str(e) or e.__class__.__name__})
        return 1

    send({"type": "download", "done": total, "total": total})
    send({"type": "done"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
