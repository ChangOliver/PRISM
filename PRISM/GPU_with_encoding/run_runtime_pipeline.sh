#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
EXE="${BUILD_DIR}/epicap_runtime_pipeline"
CONFIG="${SCRIPT_DIR}/enc_config.json"
VIDEO_DIR="${VIDEO_DIR:-${SCRIPT_DIR}/../videos}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runtime_pipeline_output}"

if [[ $# -lt 1 ]]; then
    echo "Usage: ${0##*/} <video_filename> [--config <path>] [--output-dir <dir>] [--exe <path>] [--videos <dir>]" >&2
    exit 1
fi

VIDEO_NAME="$1"; shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --exe)
            EXE="$2"; shift 2 ;;
        --videos)
            VIDEO_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

VIDEO_PATH="${VIDEO_DIR}/${VIDEO_NAME}"

if [[ ! -x "${EXE}" ]]; then
    echo "Executable not found (run 'make' in ${SCRIPT_DIR} first): ${EXE}" >&2
    exit 1
fi

if [[ ! -f "${VIDEO_PATH}" ]]; then
    echo "Video not found: ${VIDEO_PATH}" >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/run_runtime_pipeline.py" \
    --config "${CONFIG}" \
    --video "${VIDEO_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --exe "${EXE}"
