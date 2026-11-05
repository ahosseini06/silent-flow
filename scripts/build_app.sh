#!/usr/bin/env bash
# Builds build/SilentFlow.app (no Xcode needed, just the Swift toolchain).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

swift build -c release

APP="$ROOT/build/SilentFlow.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp .build/release/SilentFlow "$APP/Contents/MacOS/SilentFlow"
cp Resources/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"

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
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSCameraUsageDescription</key><string>SilentFlow reads your lips to type what you mouth. Video never leaves this Mac.</string>
  <key>SilentFlowRoot</key><string>$ROOT</string>
</dict>
</plist>
PLIST

codesign --force --sign - "$APP" >/dev/null
echo "Built $APP"
echo "Run it with: open \"$APP\""
