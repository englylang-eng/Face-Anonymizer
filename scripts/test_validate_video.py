import mimetypes
from pathlib import Path
import requests


def main():
    p = Path(__file__).resolve().parent.parent / "test_fast_video.mp4"
    if not p.exists():
        print("missing", p)
        return
    mime = mimetypes.guess_type(p.name)[0] or "video/mp4"
    with open(p, "rb") as f:
        r = requests.post(
            "http://127.0.0.1:5000/api/validate_video",
            files={"video": (p.name, f, mime)},
        )
        print("status", r.status_code, "json", r.text)


if __name__ == "__main__":
    main()
