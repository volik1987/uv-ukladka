#!/bin/sh
# Сборка zip, установка в чистый временный профиль Blender, тест аддона.
set -e
cd "$(dirname "$0")/.."
BLENDER="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
./build.sh >/dev/null
ZIP=$(ls dist/uv_ukladka-*.zip | tail -1)
PROFILE=$(mktemp -d /tmp/uv_ukladka_profile.XXXX)
export BLENDER_USER_RESOURCES="$PROFILE"
echo "== установка $ZIP в чистый профиль"
"$BLENDER" --factory-startup --command extension install-file -r user_default -e "$ZIP"
echo "== тест"
"$BLENDER" -b --python tests/test_addon.py 2>&1 | grep -E "^  |TEST_|Traceback|Error" || true
rm -rf "$PROFILE"
