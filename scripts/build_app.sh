#!/usr/bin/env bash
# build_app.sh — produce Otis.app under dist/.
#
# How it works
# ------------
# Contents/MacOS/Otis is a tiny native launcher (scripts/launcher/
# otis_launcher.c) with the absolute path to scripts/run.sh compiled in. It
# spawns run.sh detached — stderr to ~/.otis/launch.log — and exits.
#
# It replaces an osacompile AppleScript applet that did the same thing via
# `do shell script`, but spent 4-14 s in the AppleScript runtime before run.sh
# even started (measured 2026-10-20..25; a bare applet: 0.7-2.7 s), with
# nothing on screen meanwhile. The native launcher takes milliseconds. A
# shell-script CFBundleExecutable is not an option: Gatekeeper rejects
# non-Mach-O bundle executables on every launch path.
#
# We then write our own Info.plist (LSUIElement, bundle id, permission usage
# strings), build the icon, and ad-hoc sign the bundle.
#
# Permissions: macOS pins the Documents-folder and Microphone grants of an
# ad-hoc-signed app to its code hash. Rebuilding from the same source and
# path gives the same hash and keeps them; a different launcher, path or
# Info.plist means macOS asks once more on the next launch (Documents) and at
# the next recording (Microphone).
#
# Day-to-day workflow after a successful build:
#   1. Copy dist/Otis.app to /Applications (replace the old one).
#   2. FIRST LAUNCH ONLY:  right-click in Finder → Open → "Open" in dialog,
#      and allow Documents access when asked.
#   3. After that:  double-click, Spotlight, or Login Items all work.
#
# Re-run this script any time you've moved the project folder.
# OTIS_DIST_DIR overrides the output directory (the tests use it).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_PNG="${PROJECT_ROOT}/OtisIcon.png"
DIST_DIR="${OTIS_DIST_DIR:-${PROJECT_ROOT}/dist}"
APP_DIR="${DIST_DIR}/Otis.app"
CONTENTS="${APP_DIR}/Contents"
RES_DIR="${CONTENTS}/Resources"
RUN_SH="${PROJECT_ROOT}/scripts/run.sh"
LAUNCHER_SRC="${PROJECT_ROOT}/scripts/launcher/otis_launcher.c"
BUNDLE_ID="com.tenderstrike.otis"

color() { local c="$1"; shift; printf '\033[%sm%s\033[0m\n' "$c" "$*"; }
info()  { color "1;34" "==> $*"; }
ok()    { color "1;32" "✓  $*"; }
warn()  { color "1;33" "!  $*"; }
fatal() { color "1;31" "✗  $*"; exit 1; }

# ---------------------------------------------------------------------- macOS
[[ "$(uname -s)" == "Darwin" ]] || fatal "build_app.sh is macOS-only."

# ---------------------------------------------------------------- prereqs
[[ -f "${SOURCE_PNG}" ]]                || fatal "OtisIcon.png missing — expected at ${SOURCE_PNG}."
[[ -x "${RUN_SH}"     ]]                || fatal "scripts/run.sh missing or not executable."
[[ -f "${LAUNCHER_SRC}" ]]              || fatal "Launcher source missing — expected at ${LAUNCHER_SRC}."
command -v clang      >/dev/null 2>&1   || fatal "clang not in PATH — install the Command Line Tools: xcode-select --install"
command -v sips       >/dev/null 2>&1   || fatal "sips not in PATH (ships with macOS)."
command -v iconutil   >/dev/null 2>&1   || fatal "iconutil not in PATH (ships with macOS)."

# ---------------------------------------------------------------- venv check
if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
    warn ".venv not found at ${PROJECT_ROOT}/.venv."
    warn "The app will fail to start Otis until you run scripts/setup.sh once."
fi

# ---------------------------------------------------------------- clean
info "Building Otis.app at ${APP_DIR} ..."
rm -rf "${APP_DIR}"
mkdir -p "${CONTENTS}/MacOS" "${RES_DIR}"

# ---------------------------------------------------------------- launcher
info "Compiling the launcher ..."
clang -O2 -Wall -Werror -arch arm64 -arch x86_64 -mmacosx-version-min=13.0 \
    -DOTIS_RUN_SH="\"${RUN_SH}\"" \
    -o "${CONTENTS}/MacOS/Otis" "${LAUNCHER_SRC}"
ok "Launcher compiled (runs ${RUN_SH})."

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
    <key>CFBundleIdentifier</key>            <string>${BUNDLE_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key> <string>6.0</string>
    <key>CFBundleName</key>                  <string>Otis</string>
    <key>CFBundlePackageType</key>           <string>APPL</string>
    <key>CFBundleShortVersionString</key>    <string>0.2.0</string>
    <key>CFBundleVersion</key>               <string>2</string>
    <key>LSMinimumSystemVersion</key>        <string>13.0</string>

    <!-- Menu-bar app: don't show in Dock or app switcher. -->
    <key>LSUIElement</key>                   <true/>

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

# ---------------------------------------------------------------- PkgInfo
printf 'APPL????' > "${CONTENTS}/PkgInfo"
ok "Wrote ${CONTENTS}/PkgInfo"

# ---------------------------------------------------------------- xattr scrub
info "Stripping locally-built attributes ..."
/usr/bin/xattr -cr "${APP_DIR}" 2>/dev/null || true
ok "xattrs cleaned"

# ---------------------------------------------------------------- ad-hoc signing
info "Ad-hoc signing the bundle ..."
/usr/bin/codesign --force --deep --sign - "${APP_DIR}" >/dev/null 2>&1 \
    || fatal "codesign failed."
# A bundle whose seal does not verify (a file added or removed after signing)
# fails in confusing ways later; refuse to hand one out.
/usr/bin/codesign --verify --deep --strict "${APP_DIR}" \
    || fatal "The signed bundle does not verify."
ok "Ad-hoc signed and verified."

if /usr/bin/file "${CONTENTS}/MacOS/Otis" | grep -q "Mach-O"; then
    ok "Bundle executable is Mach-O ($(file "${CONTENTS}/MacOS/Otis" | awk -F': ' '{print $2}' | head -1))"
else
    fatal "Bundle executable is NOT Mach-O. Gatekeeper will reject. Aborting."
fi

# ---------------------------------------------------------------- final notes
echo
ok "Otis.app built at: ${APP_DIR}"
echo
info "Next steps:"
echo "  1. Quit Otis if it is running, then replace the installed copy:"
echo "       rm -rf /Applications/Otis.app && cp -R \"${APP_DIR}\" /Applications/"
echo
echo "  2. FIRST LAUNCH ONLY — in Finder, right-click /Applications/Otis.app → 'Open'."
echo "     Allow access to the Documents folder when macOS asks, and the"
echo "     microphone at the first recording."
echo
echo "  3. Login Items keep working: they point at /Applications/Otis.app."
echo
warn "Re-run scripts/build_app.sh whenever you move the project folder."
