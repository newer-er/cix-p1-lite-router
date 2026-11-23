#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode="${1:-replay}"
if (($#)); then shift; fi
case "$mode" in
  replay)
    exec "${CIX_PREVIEW_PYTHON:-python3}" -B server.py --replay "$@"
    ;;
  board)
    exec "${CIX_BOOTSTRAP_PYTHON:-python3.12}" -B board/launch.py "$@"
    ;;
  -h|--help)
    printf '%s\n' './deploy.sh replay [--port 18812]' \
      './deploy.sh board --m47b-bundle PATH --intent-assets PATH [--port 18812]'
    ;;
  *)
    printf '未知模式 %s。请运行 ./deploy.sh --help。\n' "$mode" >&2
    exit 2
    ;;
esac
