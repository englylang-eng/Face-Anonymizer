import time
from pathlib import Path

import cv2
import numpy as np
import requests


def make_test_video(path: Path, w: int = 640, h: int = 480, fps: int = 24, frames: int = 120) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, f"Frame {i}", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (100 + (i % 200), 100), (180 + (i % 200), 180), (200, 200, 200), -1)
        vw.write(img)
    vw.release()
    return path


def run_one(path: Path, fast: bool) -> float:
    url = "http://127.0.0.1:5000/api/anonymize_video"
    with open(path, "rb") as f:
        t0 = time.time()
        r = requests.post(
            url,
            files={"video": f},
            data={"type": "blur", "intensity": "30", "fast_mode": "1" if fast else "0"},
        )
        dt = time.time() - t0
    print(f"fast_mode={'1' if fast else '0'} status={r.status_code} time={dt:.3f}s")
    return dt


def main():
    p = Path(__file__).resolve().parent.parent / "test_fast_video.mp4"
    make_test_video(p)
    t_normal = run_one(p, fast=False)
    t_fast = run_one(p, fast=True)
    speedup = (t_normal / t_fast) if t_fast > 0 else 0.0
    print(f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
