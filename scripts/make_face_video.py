import sys
import cv2
import numpy as np
import urllib.request


URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"


def load_face_image() -> np.ndarray:
    with urllib.request.urlopen(URL) as resp:
        data = resp.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to load face image from URL")
    return img


def main(out_path: str) -> None:
    base = load_face_image()
    h, w = 240, 320
    frame0 = cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, 15, (w, h))
    if not vw.isOpened():
        raise RuntimeError("Failed to open VideoWriter")
    for i in range(60):
        # subtle horizontal shift to simulate motion
        dx = int((i % 20) - 10)
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        frame = cv2.warpAffine(frame0, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        vw.write(frame)
    vw.release()
    print(out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m scripts.make_face_video <output_path>")
        sys.exit(2)
    main(sys.argv[1])
