#!/usr/bin/env python3
import argparse
import json
import subprocess
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run GPU runtime pipeline on ladder variants.")
    parser.add_argument("--config", default="enc_config.json",
                        help="Path to ladder configuration JSON")
    parser.add_argument("--video", required=True, help="Source video path")
    parser.add_argument("--output-dir", default="runtime_pipeline_output",
                        help="Directory for detail CSVs and summary")
    parser.add_argument("--exe", default="build/epicap_runtime_pipeline",
                        help="Path to epicap_runtime_pipeline executable")
    return parser.parse_args()


def ffprobe_fps(video: Path) -> int:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video)
    ]
    output = subprocess.check_output(cmd, text=True).strip()
    if "/" in output:
        num, den = output.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(output)
    return max(1, int(round(fps)))


def main():
    args = parse_args()
    config_path = Path(args.config)
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    exe_path = Path(args.exe)

    if not exe_path.exists():
        raise SystemExit(f"Executable not found: {exe_path}")
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cfg = json.loads(config_path.read_text())
    ladder = cfg.get("ladder", [])
    if not ladder:
        raise SystemExit("Config ladder list is empty")

    options = cfg.get("options", {})
    screen_size = int(options.get("size_inches", 14))
    viewing_distance = int(options.get("viewing_distance", 24))
    hdr = bool(options.get("hdr", False))
    preset = options.get("preset", "")
    crf = options.get("crf", None)

    codec_from_config = None
    if ladder:
        codec_val = ladder[0].get("codec", "h264")
        codec_map = {
            "h264": "libx264",
            "avc": "libx264",
        }
        codec_from_config = codec_map.get(str(codec_val).lower(), str(codec_val))

    fps = ffprobe_fps(video_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "runtime_summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    cmd = [
        str(exe_path),
        "--video", str(video_path),
        "--summary", str(summary_path),
        "--fps", str(fps),
        "--size", str(screen_size),
        "--distance", str(viewing_distance),
    ]
    if hdr:
        cmd.append("--hdr")
    if preset:
        cmd.extend(["--preset", str(preset)])
    if crf is not None:
        cmd.extend(["--crf", str(int(crf))])
    if codec_from_config:
        cmd.extend(["--codec", codec_from_config])

    video_name = video_path.stem
    for entry in ladder:
        name = entry.get("name")
        width = int(entry.get("width"))
        height = int(entry.get("height"))
        if not name:
            continue
        bitrate = int(entry.get("bitrate", 0))
        detail_path = output_dir / f"{video_name}_{name}_detail.csv"
        if detail_path.exists():
            detail_path.unlink()
        variant_spec = f"{name}:{width}:{height}:{bitrate}:{detail_path}"
        cmd.extend(["--variant", variant_spec])

    subprocess.run(cmd, check=True)

    print(f"[SUCCESS] Summary written to {summary_path}")
    print(f"[SUCCESS] Timings written to {output_dir / 'timings.csv'}")
    print(f"[SUCCESS] Detail CSVs in {output_dir}")


if __name__ == "__main__":
    main()
