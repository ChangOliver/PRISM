#!/bin/bash

# GPU baseline runner for artifact (summary only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
VIDEO_ROOT="${VIDEO_ROOT:-${SCRIPT_DIR}/../videos}"
OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/results_gpu.csv}"

DURATIONS=("5s" "10s" "60s" "300s" "600s" "1800s")
RESOLUTIONS=(
    "720:720:1280"
    "1080:1080:1920"
    "2000:1440:2560"
)
DEFAULT_SIZE="${DEFAULT_SIZE:-14}"
DEFAULT_DEPTH="${DEFAULT_DEPTH:-24}"

print_status() { echo "[INFO] $1"; }
print_success() { echo "[SUCCESS] $1"; }
print_error() { echo "[ERROR] $1"; }

print_status "Video directory: ${VIDEO_ROOT}"
if [[ ! -x "${BUILD_DIR}/epicap_gpu_optimized" ]]; then
    print_error "GPU binary not found. Run 'make' in ${SCRIPT_DIR} first."
    exit 1
fi

process_video() {
    local input_file=$1
    local width=$2
    local height=$3

    if [[ ! -f "$input_file" ]]; then
        print_status "Skipping $input_file (not found)"
        return
    fi

    print_status "Processing $input_file (${width}x${height})"
    "${BUILD_DIR}/epicap_gpu_optimized" \
        -in "$input_file" \
        -h "$height" \
        -w "$width" \
        -size "$DEFAULT_SIZE" \
        -d "$DEFAULT_DEPTH" \
        -out "$OUTPUT_FILE"
    print_success "Completed $input_file"
}

print_status "Starting standard resolution loop..."
for duration in "${DURATIONS[@]}"; do
    process_video "${VIDEO_ROOT}/${duration}.mp4" 576 1024

done

print_status "Starting multi-resolution loop..."
for duration in "${DURATIONS[@]}"; do
    for resolution in "${RESOLUTIONS[@]}"; do
        IFS=':' read -r res_name width height <<<"${resolution}"
        process_video "${VIDEO_ROOT}/${duration}_${res_name}.mp4" "$width" "$height"
    done

done

print_success "All processing completed! Baseline stored in ${OUTPUT_FILE}"
