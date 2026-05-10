#!/usr/bin/env bash
# build_app.sh — produce a Gatekeeper-friendly Otis.app under dist/.
#
# How it works
# ------------
# We use ``osacompile`` (ships with macOS) to create the bundle from a tiny
# AppleScript that ``do shell script``s our launcher. ``osacompile`` produces
# a real Mach-O stub binary inside Contents/MacOS/, which Gatekeeper accepts
# for locally-built apps after a single right-click → Open. A previous
# iteration of this script wrote a bash launcher directly — that approach
# is *not* Mach-O, so Gatekeeper silently rejected every launch path
# (double-click, Spotlight, ``open``).
#
# We then:
#   * overwrite the AppleScript bundle's Info.plist with our own (LSUIElement,
#     CFBundleIdentifier, NSMicrophoneUsageDescription, etc.),
#   * replace the generic AppleScript droplet icon with Otis.icns,
#   * ad-hoc re-sign the result so the modified bundle still verifies.
#
# Day-to-day workflow after a successful build:
#   1. Drag dist/Otis.app to /Applications.
#   2. FIRST LAUNCH ONLY:  right-click in Finder → Open → "Open" in dialog.
#   3. After that:  double-click, Spotlight, or Login Items all work.
#
# Re-run this script any time you've moved the project or want a fresh build.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_PNG="${PROJECT_ROOT}/OtisIcon.png"
DIST_DIR="${PROJECT_ROOT}/dist"
APP_DIR="${DIST_DIR}/Otis.app"
CONTENTS="${APP_DIR}/Contents"
RES_DIR="${CONTENTS}/Resources"
RUN_SH="${PROJECT_ROOT}/scripts/run.sh"
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
command -v osacompile >/dev/null 2>&1   || fatal "osacompile not in PATH (ships with macOS)."
command -v sips       >/dev/null 2>&1   || fatal "sips not in PATH (ships with macOS)."
command -v iconutil   >/dev/null 2>&1   || fatal "iconutil not in PATH (ships with macOS)."

# ---------------------------------------------------------------- venv check
if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
    warn ".venv not found at ${PROJECT_ROOT}/.venv."
    warn "The launcher will fail until you run scripts/setup.sh once."
fi

# ---------------------------------------------------------------- clean
info "Building Otis.app at ${APP_DIR} ..."
rm -rf "${APP_DIR}"
mkdir -p "${DIST_DIR}"

# ---------------------------------------------------------------- compile .applescript → .app
# The AppleScript spawns our launcher and exits immediately. The launcher then
# runs Python in the background. We don't want osascript hanging around.
TMP_SCPT="$(mktemp -t otis-launcher).applescript"
cat > "${TMP_SCPT}" <<APPLESCRIPT
do shell script "exec '${RUN_SH}' >/dev/null 2>&1 &"
APPLESCRIPT

osacompile -o "${APP_DIR}" "${TMP_SCPT}"
rm -f "${TMP_SCPT}"
ok "AppleScript bundle compiled."

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
# Replace the default AppleScript droplet icon at its expected name too —
# the compiled Info.plist still references "applet.icns" by default.
cp "${RES_DIR}/Otis.icns" "${RES_DIR}/applet.icns"
rm -rf "$(dirname "${ICONSET}")"
ok "Wrote ${RES_DIR}/Otis.icns + applet.icns"

# ---------------------------------------------------------------- Info.plist
# Overwrite osacompile's default plist with our own — it doesn't include
# LSUIElement, our bundle id, or the macOS permission usage strings.
info "Writing Info.plist ..."
cat > "${CONTENTS}/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>     <string>en</string>
    <key>CFBundleDisplayName</key>           <string>Otis</string>
    <key>CFBundleExecutable</key>            <string>applet</string>
    <key>CFBundleIconFile</key>              <string>Otis</string>
    <key>CFBundleIdentifier</key>            <string>${BUNDLE_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key> <string>6.0</string>
    <key>CFBundleName</key>                  <string>Otis</string>
    <key>CFBundlePackageType</key>           <string>APPL</string>
    <key>CFBundleShortVersionString</key>    <string>0.1.0</string>
    <key>CFBundleVersion</key>               <string>1</string>
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

    <!-- AppleScript stub metadata that osacompile would have emitted. -->
    <key>LSRequiresCarbon</key>              <true/>
    <key>OSAAppletStayOpen</key>             <false/>
    <key>WindowState</key>                   <dict>
        <key>name</key><string>ScriptWindowState</string>
    </dict>
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
# Re-sign — we modified Resources/, so the existing Apple-tool signature
# would no longer verify.
info "Ad-hoc signing the bundle ..."
if /usr/bin/codesign --force --deep --sign - "${APP_DIR}" >/dev/null 2>&1; then
    ok "Ad-hoc signed."
else
    warn "codesign failed — first launch may show 'unidentified developer'."
fi

# Verify the executable inside is a real Mach-O (not a script — that was the
# old bug). If this fails, Gatekeeper will reject every launch path.
if /usr/bin/file "${CONTENTS}/MacOS/applet" | grep -q "Mach-O"; then
    ok "Bundle executable is Mach-O ($(file "${CONTENTS}/MacOS/applet" | awk -F': ' '{print $2}'))"
else
    fatal "Bundle executable is NOT Mach-O. Gatekeeper will reject. Aborting."
fi

# ---------------------------------------------------------------- final notes
echo
ok "Otis.app built at: ${APP_DIR}"
echo
info "Next steps:"
echo "  1. mv ${APP_DIR} /Applications/"
echo
echo "  2. FIRST LAUNCH ONLY — bypass Gatekeeper:"
echo "       In Finder, right-click /Applications/Otis.app → 'Open'."
echo "       Click 'Open' in the dialog that appears."
echo "     (One-time per build. After this, double-click and Spotlight work.)"
echo
echo "  3. Subsequent launches: ⌘+Space → \"Otis\" → ⏎,  or double-click."
echo
echo "  4. (Auto-launch at login) System Settings → General → Login Items"
echo "     → \"+\" → /Applications/Otis.app"
echo
info "First launch will trigger macOS permission prompts:"
echo "  • Microphone — required for recording"
echo "  • Apple Events / Notifications — required for menu-bar UI"
echo
warn "Re-run scripts/build_app.sh whenever you move the project folder."
