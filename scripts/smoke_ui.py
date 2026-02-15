import os
import sys
import time
import json
import mimetypes
import tempfile
from pathlib import Path

try:
    import requests  # type: ignore
except Exception:
    requests = None

import cv2
import numpy as np

BASE = "http://127.0.0.1:5000"

def http_get(path: str):
    url = BASE + path
    if requests:
        r = requests.get(url)
        return r.status_code, r.text, r.headers
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(url) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            headers = dict(resp.info())
            return resp.getcode(), body, headers
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="ignore"), dict(e.headers or {})

def http_post_files(path: str, files: dict, data: dict = None):
    url = BASE + path
    data = data or {}
    if not requests:
        raise RuntimeError("requests not available for multipart POST")
    r = requests.post(url, files=files, data=data)
    ct = r.headers.get("content-type", "")
    body = r.text if ("text" in ct or "json" in ct) else ""
    return r.status_code, body, r.headers

def make_test_image(tmpdir: Path) -> Path:
    img = np.full((240, 240, 3), 255, dtype=np.uint8)
    cv2.circle(img, (120, 120), 60, (0, 0, 0), 3)
    cv2.circle(img, (100, 110), 8, (0, 0, 0), -1)
    cv2.circle(img, (140, 110), 8, (0, 0, 0), -1)
    cv2.ellipse(img, (120, 140), (25, 12), 0, 0, 180, (0, 0, 0), 2)
    path = tmpdir / "test.jpg"
    cv2.imwrite(str(path), img)
    return path

def make_test_video(tmpdir: Path) -> Path:
    path = tmpdir / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (320, 240))
    for i in range(48):
        frame = np.full((240, 320, 3), 255, dtype=np.uint8)
        cv2.putText(frame, f"Frame {i+1}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        writer.write(frame)
    writer.release()
    return path

def check_ui_elements(html: str):
    ids = [
        "tabImage","tabVideo","tabImageMobile","tabVideoMobile",
        "dropArea","statusText","fileInput","downloadBtn",
        "typeSelectBtn","typeMenu","intensityRange","fastModeToggle",
        "anonymizeBtn","clearBtn","resultBox","toastContainer",
    ]
    missing = [i for i in ids if (f'id="{i}"' not in html)]
    return missing

def main():
    results = {"ok": True, "checks": []}

    code, body, headers = http_get("/")
    results["checks"].append({"name": "GET /", "status": code})
    if code != 200:
        results["ok"] = False

    code, html, headers = http_get("/web/index.html")
    results["checks"].append({"name": "GET /web/index.html", "status": code})
    if code == 200:
        miss = check_ui_elements(html)
        results["checks"].append({"name": "UI elements present", "status": 200 if not miss else 500, "missing": miss})
        if miss:
            results["ok"] = False
    else:
        results["ok"] = False

    with tempfile.TemporaryDirectory() as td:
        tdpath = Path(td)
        img = make_test_image(tdpath)
        mime = mimetypes.guess_type(img.name)[0] or "image/jpeg"
        status, body, hdrs = http_post_files("/api/validate_image", files={"image": (img.name, open(img, "rb"), mime)})
        results["checks"].append({"name": "POST /api/validate_image", "status": status})
        status, body, hdrs = http_post_files("/api/anonymize", files={"image": (img.name, open(img, "rb"), mime)}, data={"type": "blur", "intensity": "30"})
        results["checks"].append({"name": "POST /api/anonymize (expect 200 or 400)", "status": status})

        vid = make_test_video(tdpath)
        vmime = mimetypes.guess_type(vid.name)[0] or "video/mp4"
        status, body, hdrs = http_post_files("/api/validate_video", files={"video": (vid.name, open(vid, "rb"), vmime)})
        results["checks"].append({"name": "POST /api/validate_video", "status": status})

        status, body, hdrs = http_post_files("/api/start_job_video", files={"video": (vid.name, open(vid, "rb"), vmime)}, data={"type": "blur", "intensity": "30", "fast_mode": "1"})
        results["checks"].append({"name": "POST /api/start_job_video", "status": status})
        job_id = ""
        try:
            j = json.loads(body)
            if j.get("ok"):
                job_id = j.get("id", "")
        except Exception:
            pass
        if job_id:
            for _ in range(60):
                sc, jb, _ = http_get(f"/api/job_progress?id={job_id}")
                results["checks"].append({"name": "GET /api/job_progress", "status": sc})
                if sc == 200:
                    try:
                        jbj = json.loads(jb)
                        if jbj.get("done"):
                            break
                    except Exception:
                        pass
                time.sleep(0.5)
            sc, _, hdr = http_get(f"/api/job_download?id={job_id}")
            results["checks"].append({"name": "GET /api/job_download", "status": sc})
            if sc != 200:
                results["ok"] = False
        else:
            results["ok"] = False

    print(json.dumps(results, indent=2))
    return 0 if results["ok"] else 1

if __name__ == "__main__":
    sys.exit(main())
