#!/bin/sh
# Сборка установочного zip аддона (Blender extension) в dist/
set -e
cd "$(dirname "$0")"
BLENDER="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
find uv_ukladka -name __pycache__ -type d -exec rm -rf {} +
mkdir -p dist
"$BLENDER" --command extension validate uv_ukladka
"$BLENDER" --command extension build --source-dir uv_ukladka --output-dir dist
ls -la dist
