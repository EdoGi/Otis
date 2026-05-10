#!/usr/bin/env bash
# build_app.sh — produce a hand-crafted Otis.app under dist/.
#
# What you get:
#   dist/
#     └─ Otis.app/
#        └─ Contents/
#           ├─ Info.plist            (CFBundleName, LSUIElement, NSMicrophoneUsageDescription…)
#           ├─ MacOS/
#           │   └─ Otis              (executable launcher: cd $PROJECT && source .venv && exec python -m src.main)
#           └─ Resources/
#               ├─ Otis.icns         (multi-resolution icon converted from OtisIcon.png)
#               └─ OtisIcon.png      (raw source, kept for reference)
#
# Day-to-day workflow after a successful build:
#   1. Drag dist/Otis.app to /Applications.
#   2. Double-click it (or Spotlight: ⌘+Space → "Otis" ⏎).
#   3. To auto-launch at login: System Settings → General → Login Items → "+" → choose Otis.
#
# Re-run this script any time you've moved the project or want a fresh build.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_PNG="${PROJECT_ROOT}/OtisIcon.png"
DIST_DIR="${PROJECT_ROOT}/dist"
APP_DIR="${DIST_DIR}/Otis.app"
CONTENTS="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS}/MacOS"
RES_DIR="${CONTENTS}/Resources"
LAUNCHER="${MACOS_DIR}/Otis"

color() { local c="$1"; shift; printf '\033[%sm%s\033[0m\n' "$c" "$*"; }
info()  { color "1;34" "==> $*"; }
ok()    { color "1;32" "✓  $*"; }
warn()  { color "1;33" "!  $*"; }
fatal() { color "1;31" "✗  $*"; exit 1; }

# ---------------------------------------------------------------------- macOS
[[ "$(uname -s)" == "Darwin" ]] || fatal "build_app.sh is macOS-only."

# ---------------------------------------------------------------- prereqs
[[ -f "${SOURCE_PNG}" ]] || fatal "OtisIcon.png missing — expected at ${SOURCE_PNG}."
command -v sips     >/dev/null 2>&1 || fatal "sips not in PATH (ships with macOS)."
command -v iconutil >/dev/null 2>&1 || fatal "iconutil not in PATH (ships with macOS)."

# ---------------------------------------------------------------- venv check
if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
    warn ".venv not found at ${PROJECT_ROOT}/.venv."
    warn "The launcher will fail until you run scripts/setup.sh once."
fi

# ---------------------------------------------------------------- clean / mkdir
info "Building Otis.app at ${APP_DIR} ..."
rm -rf "${APP_DIR}"
mkdir -p "${MACOS_DIR}" "${RES_DIR}"

# ---------------------------------------------------------------- iconset → icns
info "Generating multi-resolution icon (.icns) from OtisIcon.png ..."
ICONSET="$(mktemp -d)/Otis.iconset"
mkdir -p "${ICONSET}"
for entry in \
    "16  icon_16x16.png" \
    "32  icon_16x16@2x.png" \
    "32  icon_32x32.png" \
    "64  icon_32x32@2x.png" \
    "128 icon_128x128.png" \
    "256 icon_128x128@2x.png" \
    "256 icon_256x256.png" \
    "512 icon_256x256@2x.png" \
    "512 icon_512x512.png" \
    "1024 icon_512x512@2x.png"; do
    size="${entry%% *}"
    name="${entry##* }"
    sips -z "${size}" "${size}" "${SOURCE_PNG}" --out "${ICONSET}/${name}" >/dev/null
done
iconutil -c icns "${ICONSET}" -o "${RES_DIR}/Otis.icns"
cp "${SOURCE_PNG}" "${RES_DIR}/OtisIcon.png"
rm -rf "$(dirname "${ICONSET}")"
ok "Wrote ${RES_DIR}/Otis.icns"

# ---------------------------------------------------------------- Info.plist
info "Writing Info.plist ..."
cat > "${CONTENTS}/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>     <string>en</string>
    <key>CFBundleDisplayName</key>           <string>Otis</string>
    <key>CFBundleExecutable</key>            <string>Otis</string>
    <key>CFBundleIconFile</key>              <string>Otis</string>
    <key>CFBundleIdentifier</key>            <string>com.tenderstrike.otis</string>
    <key>CFBundleInfoDictionaryVersion</key> <string>6.0</string>
    <key>CFBundleName</key>                  <string>Otis</string>
    <key>CFBundlePackageType</key>           <string>APPL</string>
    <key>CFBundleShortVersionString</key>    <string>0.1.0</string>
    <key>CFBundleVersion</key>               <string>1</string>
    <key>LSMinimumSystemVersion</key>        <string>13.0</string>

    <!-- Menu-bar app: don't show in Dock or app switcher. -->
    <key>LSUIElement</key>                   <true/>
    <!-- Only one Otis at a time — clicking the .app while it's running
         brings the existing instance forward instead of starting a second. -->
    <key>LSMultipleInstancesProhibited</key> <true/>

    <!-- Permission usage strings — without these, macOS denies silently. -->
    <key>NSMicrophoneUsageDescription</key>
    <string>Otis needs microphone access to record meetings you initiate.</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>Otis uses Apple Events to detect sleep/wake and pause recording.</string>
    <key>NSCalendarsUsageDescription</key>
    <string>Otis can use your calendar to alert you before a meeting starts.</string>
</dict>
</plist>
EOF
ok "Wrote ${CONTENTS}/Info.plist"

# ---------------------------------------------------------------- launcher
info "Writing launcher script ..."
cat > "${LAUNCHER}" <<EOF
#!/usr/bin/env bash
# Auto-generated by scripts/build_app.sh.
# Do not edit — re-run the build script to regenerate.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT}"

if [[ ! -d "\${PROJECT_ROOT}" ]]; then
    /usr/bin/osascript -e 'display alert "Otis" message "Project folder not found at: '"\${PROJECT_ROOT}"'\n\nMove or rebuild the .app via scripts/build_app.sh."'
    exit 1
fi

cd "\${PROJECT_ROOT}"

if [[ ! -d ".venv" ]]; then
    /usr/bin/osascript -e 'display alert "Otis" message "Python venv missing.\n\nIn Terminal:\n  cd '"\${PROJECT_ROOT}"' && ./scripts/setup.sh"'
    exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

# Logs land in ~/.otis/otis.log via the in-app file handler — but if
# something blows up before logging is set up we still want a trail.
LOG_DIR="\${HOME}/.otis"
mkdir -p "\${LOG_DIR}"
exec >> "\${LOG_DIR}/launch.log" 2>&1
echo "------ Otis.app launching at \$(date) ------"

exec python -m src.main "\$@"
EOF
chmod +x "${LAUNCHER}"
ok "Wrote ${LAUNCHER} (chmod +x)"

# ---------------------------------------------------------------- final notes
echo
ok "Otis.app built at: ${APP_DIR}"
echo
info "Next steps:"
echo "  1. mv ${APP_DIR} /Applications/"
echo "  2. Open it from Spotlight (⌘+Space → \"Otis\")."
echo "  3. (Auto-launch) System Settings → General → Login Items → \"+\" → /Applications/Otis.app"
echo
info "First launch will trigger macOS permission prompts:"
echo "  • Microphone — required for recording"
echo "  • Apple Events / Notifications — required for menu-bar UI"
echo
warn "Re-run scripts/build_app.sh whenever you move the project folder."
