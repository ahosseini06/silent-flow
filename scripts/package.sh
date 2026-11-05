#!/usr/bin/env bash
# Packages SilentFlow into a disk image anyone can install: build/SilentFlow-<version>.dmg
#
#   ./scripts/package.sh
#
# Unlike ./scripts/build_app.sh (a developer build that runs from this checkout), the packaged app
# carries everything except the model weights:
#   Contents/MacOS/SilentFlow                 the Swift app
#   Contents/Resources/inference/             the sidecar (server.py, fetch_models.py, espnet/ …)
#   Contents/Resources/runtime/python/        a relocatable CPython 3.12 with the locked deps
#                                             (torch, mlx-lm, …) installed into it
# The weights (~4.3 GB) are too big for a GitHub release file, so on first launch the app runs
# inference/fetch_models.py and downloads them into ~/Library/Application Support/SilentFlow/models.
#
# Needs: Apple Silicon, the Swift toolchain, and uv (brew install uv).
# Ad-hoc signed: on first open, macOS asks the user to approve it once (see the note in the image).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="0.1.0"
PY="3.12"
APP="$ROOT/build/SilentFlow.app"
DMG="$ROOT/build/SilentFlow-$VERSION.dmg"
command -v uv >/dev/null || { echo "uv is required: brew install uv"; exit 1; }

echo "==> Swift app"
swift build -c release
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp .build/release/SilentFlow "$APP/Contents/MacOS/SilentFlow"
cp Resources/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"

echo "==> Python $PY runtime"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
uv python install "$PY" --install-dir "$WORK/pythons" --no-bin >/dev/null
SRC_PY="$(find "$WORK/pythons" -maxdepth 1 -type d -name "cpython-$PY*-macos-aarch64-none" | head -1)"
[[ -n "$SRC_PY" ]] || { echo "couldn't find the downloaded CPython $PY"; exit 1; }
RT="$APP/Contents/Resources/runtime/python"
mkdir -p "$(dirname "$RT")"
cp -R "$SRC_PY" "$RT"
rm -f "$RT"/lib/python*/EXTERNALLY-MANAGED   # this copy belongs to the app; let uv install into it

echo "==> Locked dependencies"
(cd inference && uv export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt) > "$WORK/requirements.txt"
uv pip install --python "$RT/bin/python3" --no-cache --compile-bytecode -r "$WORK/requirements.txt"
# Trim what the app never imports.
rm -rf "$RT"/lib/python*/{test,idlelib,tkinter,turtledemo,ensurepip} "$RT"/lib/{tcl,tk}* "$RT/share"
find "$RT" -name "tests" -type d -path "*/site-packages/*" -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$RT"/lib/python*/site-packages/torch/include

echo "==> Sidecar"
mkdir -p "$APP/Contents/Resources/inference"
rsync -a --exclude ".venv" --exclude "__pycache__" --exclude "tools" --exclude "uv.lock" \
  inference/ "$APP/Contents/Resources/inference/"
"$RT/bin/python3" -m compileall -q "$APP/Contents/Resources/inference" >/dev/null || true

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>SilentFlow</string>
  <key>CFBundleDisplayName</key><string>SilentFlow</string>
  <key>CFBundleIdentifier</key><string>local.silentflow.app</string>
  <key>CFBundleExecutable</key><string>SilentFlow</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>${GITHUB_RUN_NUMBER:-1}</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSCameraUsageDescription</key><string>SilentFlow reads your lips to type what you mouth. Video never leaves this Mac.</string>
</dict>
</plist>
PLIST

echo "==> Signing (ad-hoc)"
# Nested code first: every Mach-O in the runtime, then the bundle itself.
find "$APP/Contents/Resources/runtime" -type f \( -name "*.so" -o -name "*.dylib" -o -perm -u+x \) -print0 |
  while IFS= read -r -d '' f; do
    if file -b "$f" | grep -q "Mach-O"; then codesign --force --sign - "$f" 2>/dev/null || true; fi
  done
codesign --force --sign - "$APP"
codesign --verify "$APP"

echo "==> Disk image"
STAGE="$WORK/stage"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/SilentFlow.app"
ln -s /Applications "$STAGE/Applications"
cat > "$STAGE/Read me first.txt" <<'NOTE'
SilentFlow, installing

1. Drag SilentFlow into Applications.
2. Open it. macOS will say it can't verify the developer, because this build isn't notarised by
   Apple yet. Click Done.
3. Open System Settings → Privacy & Security, scroll down to "SilentFlow was blocked…", click
   Open Anyway, and confirm. You only do this once.

   (Or, in Terminal:  xattr -dr com.apple.quarantine /Applications/SilentFlow.app )

On first launch SilentFlow downloads its lip-reading models (about 4.3 GB) from Hugging Face into
~/Library/Application Support/SilentFlow. After that it works offline: no audio is ever recorded,
and nothing you mouth leaves your Mac.

Needs a Mac with Apple Silicon running macOS 15 or later, and a webcam.
NOTE
rm -f "$DMG"
hdiutil create -volname "SilentFlow $VERSION" -srcfolder "$STAGE" -ov -format UDZO -fs HFS+ "$DMG" >/dev/null

echo "Packaged $DMG ($(du -h "$DMG" | cut -f1); app $(du -sh "$APP" | cut -f1))"
