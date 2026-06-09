#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="Create MP4 from rendered PNG sequence.")
    p.add_argument("--input", required=True, help="Folder containing frame_000000.png")
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=16)
    args = p.parse_args()

    inp = Path(args.input).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pattern = inp / "frame_%06d.png"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-framerate", str(args.fps),
            "-i", str(pattern),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(args.crf),
            str(out),
        ]
        print("[make_video]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"[make_video] wrote {out}")
        return 0

    try:
        import cv2  # type: ignore
    except Exception:
        print("[make_video] ffmpeg not found and OpenCV is unavailable.", file=sys.stderr)
        return 1

    images = sorted(inp.glob("frame_*.png"))
    if not images:
        print(f"[make_video] no PNG frames found in {inp}", file=sys.stderr)
        return 1
    first = cv2.imread(str(images[0]))
    if first is None:
        print(f"[make_video] cannot read {images[0]}", file=sys.stderr)
        return 1
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for img in images:
        frame = cv2.imread(str(img))
        if frame is None:
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
    writer.release()
    print(f"[make_video] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
