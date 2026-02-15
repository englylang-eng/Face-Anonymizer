import os
import urllib.request
from pathlib import Path


def main() -> None:
    base = Path(__file__).resolve().parent.parent / "models"
    base.mkdir(parents=True, exist_ok=True)
    files = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    }
    for name, url in files.items():
        out = base / name
        print(f"downloading {url} -> {out}")
        urllib.request.urlretrieve(url, str(out))
    print("done")


if __name__ == "__main__":
    main()
