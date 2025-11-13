# EpiCap Artifact: CPU/GPU Baselines and In‑Encoder Pipeline

This artifact packages three reproducible implementations of the TBD:

- CPU_version: formally verified CPU baseline (summary only).
- GPU_version: standalone, fastest CUDA implementation (summary only).
- GPU_with_encoding: decode‑once pipeline that resizes on GPU, re‑encodes each rung (libx264), and runs the detector on reconstructed frames (what viewers actually see). Produces per‑frame detail and per‑stage timings.

All components use the bundled sample videos and a shared inverse‑gamma lookup table.

--------------------------------------------------------------------------------
## 1. Directory Layout

```
artifact/
├── CPU_version/
│   ├── CMakeLists.txt, Makefile, main.cpp, utils.*
│   └── run_cpu_bash.sh     # generates results.csv
├── GPU_version/
│   ├── CMakeLists.txt, Makefile, gpu_optimized_app.* , main_gpu_correct.cu
│   └── run_gpu_bash.sh     # generates results_gpu.csv
├── GPU_with_encoding/
│   ├── CMakeLists.txt, include/, src/
│   ├── Makefile            # builds build/epicap_runtime_pipeline
│   ├── run_runtime_pipeline.py, run_runtime_pipeline.sh
│   └── enc_config.json     # ladder definition (base/1080/720/576)
├── videos/                 # portrait test clips (e.g., 5s_2000.mp4)
├── inverseGammaLUT.bin     # shared LUT (256 x float)
└── ep_types_gpu.h          # shared GPU types
```

--------------------------------------------------------------------------------
## 2. System Requirements

- OS: Linux (tested on recent Ubuntu).
- Build tools: CMake ≥ 3.22, GNU g++ ≥ 11.
- OpenCV ≥ 4.5 (core, imgproc, videoio) — headers and libs.
- CUDA Toolkit ≥ 11.8 (GPU_version and GPU_with_encoding only).
- FFmpeg dev libraries (GPU_with_encoding only): libavcodec, libavutil, libswscale; x264 enabled (`--enable-libx264`).

Check your FFmpeg: `ffmpeg -version` should list `--enable-libx264` and the above libs.

--------------------------------------------------------------------------------
## 3. Build Instructions

CPU baseline:
```
cd artifact/CPU_version
make            # builds build/epicap
```

GPU baseline:
```
cd artifact/GPU_version
make            # builds build/epicap_gpu_optimized
```

GPU with encoding (inline x264 + GPU detector):
```
cd artifact/GPU_with_encoding
make            # builds build/epicap_runtime_pipeline
```

Use `make clean` in any folder to wipe its build directory.

--------------------------------------------------------------------------------
## 4. Running Experiments

The videos are vertical (portrait). Width < Height in the file names:
- 2000:        1440 x 2560
- 1080:        1080 x 1920
- 720:          720 x 1280
- base:         576 x 1024

CPU baseline (summary CSV):
```
cd artifact/CPU_version
./run_cpu_bash.sh     # writes results.csv
```

GPU baseline (summary CSV):
```
cd artifact/GPU_version
./run_gpu_bash.sh     # writes results_gpu.csv
```

GPU with encoding (detail + timings):
```
cd artifact/GPU_with_encoding
./run_runtime_pipeline.sh 5s_2000.mp4
```
Outputs go to `GPU_with_encoding/runtime_pipeline_output/`:
- `<video>_<rung>_detail.csv` — per‑frame (frame,harmfulLumCount,harmfulColCount)
- `timings.csv` — per‑rung decode_ms, resize_ms, encode_ms, detection_ms
- `runtime_summary.csv` — one row per rung with flags and counts

Environment overrides for the runner: `VIDEO_DIR`, `OUTPUT_DIR` (see the script).

--------------------------------------------------------------------------------
## 5. What’s Measured

GPU_with_encoding’s `timings.csv` reports:
- decode_ms: CPU decode + host→device upload (shared; only listed once).
- resize_ms: GPU upload/resize/downscale per rung.
- encode_ms: CPU libx264 encode + flush (includes BGR→YUV and YUV→BGR recon conversion).
- detection_ms: GPU detector kernel + reduction on reconstructed frames.

The total wall‑clock used to be printed in `runtime_summary.csv`; it is intentionally removed to avoid duplication. Sum the timing columns for an apples‑to‑apples comparison.

--------------------------------------------------------------------------------
## 6. Reproducibility Notes

- Determinism: CPU/GPU detectors use fixed thresholds; results are deterministic given identical inputs and toolchain.
- Portrait coordinate convention: runners always pass height then width to match video layout.
- LUT path: CPU seeks `inverseGammaLUT.bin` in `CPU_version/` then `artifact/`; GPU loads from working directory (the file resides at artifact root).
- libx264 banners printed at startup (capabilities, profile/level) are informational.

--------------------------------------------------------------------------------
## 7. Configuration

`GPU_with_encoding/enc_config.json` controls the ladder and encoder hints (name, width, height, codec, bitrate, scale filter; global preset/crf). The pipeline always analyzes reconstructed frames to match viewer pixels.

--------------------------------------------------------------------------------
