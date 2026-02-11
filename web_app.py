import os
from io import BytesIO
import numpy as np
import cv2
from flask import Flask, request, send_file, send_from_directory, jsonify
from tempfile import NamedTemporaryFile
from anonymizer import process_array, process_video
from anonymizer import count_faces_array
from anonymizer import probe_video_faces
import mimetypes
import shutil
from flask_cors import CORS
import threading
import uuid

app = Flask(__name__, static_folder="web")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.get("/")
def index():
    resp = send_from_directory(app.static_folder, "index.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.get("/images/<path:filename>")
def images(filename):
    root = os.path.join(os.path.dirname(__file__), "images")
    resp = send_from_directory(root, filename)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.get("/images")
def images_list():
    root = os.path.join(os.path.dirname(__file__), "images")
    try:
        files = os.listdir(root)
    except Exception:
        files = []
    return jsonify({"files": files})

@app.after_request
def add_no_cache(resp):
    try:
        resp.headers.setdefault("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        resp.headers.setdefault("Pragma", "no-cache")
        resp.headers.setdefault("Expires", "0")
        csp = "; ".join([
            "default-src 'self'",
            "script-src 'self' https://cdn.tailwindcss.com 'unsafe-inline'",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "img-src 'self' data: blob:",
            "font-src 'self' data: https://fonts.gstatic.com",
            "connect-src 'self'",
            "media-src 'self' blob:",
            "object-src 'none'",
            "frame-ancestors 'self'",
            "base-uri 'self'",
            "upgrade-insecure-requests",
        ])
        resp.headers.setdefault("Content-Security-Policy", csp)
    except Exception:
        pass
    return resp
@app.get("/logo.png")
def serve_logo():
    path = os.path.join(os.path.dirname(__file__), "images", "logo.png")
    if not os.path.exists(path):
        return jsonify({"error": "logo missing"}), 404
    mime, _ = mimetypes.guess_type(path)
    return send_file(path, mimetype=mime or "application/octet-stream")

@app.get("/web/images/logo.png")
def serve_web_logo():
    path = os.path.join(os.path.dirname(__file__), "images", "logo.png")
    if not os.path.exists(path):
        return jsonify({"error": "logo missing"}), 404
    mime, _ = mimetypes.guess_type(path)
    return send_file(path, mimetype=mime or "application/octet-stream")

@app.get("/logo")
def serve_any_logo():
    base = os.path.join(os.path.dirname(__file__), "images")
    candidates = ["logo.svg", "logo.png", "logo.jpg", "logo.jpeg", "logo.webp"]
    for name in candidates:
        p = os.path.join(base, name)
        if os.path.exists(p):
            mime, _ = mimetypes.guess_type(p)
            return send_file(p, mimetype=mime or "application/octet-stream")
    return jsonify({"error": "logo missing"}), 404
@app.get("/brand/logo")
def serve_brand_logo():
    base = os.path.join(os.path.dirname(__file__), "images")
    candidates = ["logo.svg", "logo.png", "logo.jpg", "logo.jpeg", "logo.webp"]
    for name in candidates:
        p = os.path.join(base, name)
        if os.path.exists(p):
            mime, _ = mimetypes.guess_type(p)
            return send_file(p, mimetype=mime or "application/octet-stream")
    return jsonify({"error": "logo missing"}), 404
@app.get("/refresh_logo")
def refresh_logo():
    src = os.path.join(os.path.dirname(__file__), "images", "logo.png")
    dst = os.path.join(os.path.dirname(__file__), "web", "logo.png")
    if not os.path.exists(src):
        return jsonify({"ok": False, "error": "source missing"}), 404
    try:
        shutil.copyfile(src, dst)
        size = os.path.getsize(dst)
        return jsonify({"ok": True, "size": size})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
@app.post("/api/anonymize")
def anonymize():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400
    file = request.files["image"]
    ext = os.path.splitext(file.filename or "")[1].lower()
    if file.mimetype not in ("image/jpeg","image/png","image/webp","image/bmp","image/tiff"):
        return jsonify({"error": "unsupported image type"}), 400
    if ext not in (".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"):
        return jsonify({"error": "unsupported image extension"}), 400
    data = file.read()
    if not data:
        return jsonify({"error": "empty file"}), 400
    if len(data) > 10 * 1024 * 1024:
        return jsonify({"error": "image too large"}), 400
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "invalid image"}), 400
    method = request.form.get("type", "blur")
    try:
        intensity = int(request.form.get("intensity", "30"))
    except:
        intensity = 30
    processed, count = process_array(img, method=method, intensity=intensity)
    if count == 0:
        return jsonify({"error": "no faces found"}), 400
    ok, buf = cv2.imencode(".jpg", processed)
    if not ok:
        return jsonify({"error": "encode error"}), 500
    bio = BytesIO(buf.tobytes())
    bio.seek(0)
    return send_file(bio, mimetype="image/jpeg", download_name="anonymized.jpg")

@app.post("/api/validate_image")
def validate_image():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "no image"}), 400
    file = request.files["image"]
    ext = os.path.splitext(file.filename or "")[1].lower()
    if file.mimetype not in ("image/jpeg","image/png","image/webp","image/bmp","image/tiff"):
        return jsonify({"ok": False, "error": "unsupported image type"}), 400
    if ext not in (".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"):
        return jsonify({"ok": False, "error": "unsupported image extension"}), 400
    data = file.read()
    if not data:
        return jsonify({"ok": False, "error": "empty file"}), 400
    if len(data) > 10 * 1024 * 1024:
        return jsonify({"ok": False, "error": "image too large"}), 400
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "error": "invalid image"}), 400
    faces = count_faces_array(img)
    return jsonify({"ok": True, "faces": int(faces)})

@app.post("/api/anonymize_video")
def anonymize_video():
    if "video" not in request.files:
        return jsonify({"error": "no video"}), 400
    file = request.files["video"]
    ext = os.path.splitext(file.filename or "")[1].lower()
    if file.mimetype not in ("video/mp4","video/webm","video/quicktime","video/x-msvideo","video/hevc"):
        return jsonify({"error": "unsupported video type"}), 400
    if ext not in (".mp4",".webm",".mov",".avi",".hevc"):
        return jsonify({"error": "unsupported video extension"}), 400
    if not file:
        return jsonify({"error": "empty file"}), 400
    method = request.form.get("type", "blur")
    try:
        intensity = int(request.form.get("intensity", "30"))
    except:
        intensity = 30
    fast_flag = request.form.get("fast_mode", "0")
    fast = True if str(fast_flag).lower() in ("1","true","yes","on") else False
    with NamedTemporaryFile(delete=False, suffix=ext or ".mp4") as tmp_in:
        file.save(tmp_in)
        tmp_in_path = tmp_in.name
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
        out_path = tmp_out.name
    try:
        output_path, total_faces = process_video(tmp_in_path, out_path, method=method, intensity=intensity, fast=fast)
        if not output_path:
            return jsonify({"error": "processing failed"}), 500
        if total_faces == 0:
            return jsonify({"error": "no faces found"}), 400
        return send_file(output_path, mimetype="video/mp4", download_name="anonymized.mp4")
    finally:
        try:
            os.remove(tmp_in_path)
        except:
            pass
@app.post("/api/start_job_video")
def start_job_video():
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "no video"}), 400
    file = request.files["video"]
    ext = os.path.splitext(file.filename or "")[1].lower()
    if file.mimetype not in ("video/mp4","video/webm","video/quicktime","video/x-msvideo","video/hevc"):
        return jsonify({"ok": False, "error": "unsupported video type"}), 400
    if ext not in (".mp4",".webm",".mov",".avi",".hevc"):
        return jsonify({"ok": False, "error": "unsupported video extension"}), 400
    method = request.form.get("type", "blur")
    try:
        intensity = int(request.form.get("intensity", "30"))
    except:
        intensity = 30
    fast_flag = request.form.get("fast_mode", "0")
    fast = True if str(fast_flag).lower() in ("1","true","yes","on") else False
    with NamedTemporaryFile(delete=False, suffix=ext or ".mp4") as tmp_in:
        file.save(tmp_in)
        tmp_in_path = tmp_in.name
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
        out_path = tmp_out.name
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"percent": 0, "done": False, "path": None, "faces": 0}
    def _cb(p):
        JOBS[job_id]["percent"] = int(p)
    def _worker():
        try:
            output_path, total_faces = process_video(tmp_in_path, out_path, method=method, intensity=intensity, fast=fast, progress_cb=_cb)
            JOBS[job_id]["faces"] = int(total_faces)
            if output_path and os.path.exists(output_path):
                JOBS[job_id]["path"] = output_path
                JOBS[job_id]["done"] = True
            else:
                JOBS[job_id]["done"] = True
                JOBS[job_id]["path"] = None
        finally:
            try:
                os.remove(tmp_in_path)
            except:
                pass
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return jsonify({"ok": True, "id": job_id})

@app.get("/api/job_progress")
def job_progress():
    job_id = request.args.get("id", "")
    if not job_id or job_id not in JOBS:
        return jsonify({"ok": False, "error": "invalid id"}), 404
    j = JOBS[job_id]
    return jsonify({"ok": True, "percent": int(j["percent"]), "done": bool(j["done"]), "faces": int(j.get("faces", 0))})

@app.get("/api/job_download")
def job_download():
    job_id = request.args.get("id", "")
    if not job_id or job_id not in JOBS:
        return jsonify({"error": "invalid id"}), 404
    j = JOBS[job_id]
    p = j.get("path")
    if not p or not os.path.exists(p):
        return jsonify({"error": "not ready"}), 400
    return send_file(p, mimetype="video/mp4", download_name="anonymized.mp4")

JOBS = {}
@app.post("/api/validate_video")
def validate_video():
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "no video"}), 400
    file = request.files["video"]
    ext = os.path.splitext(file.filename or "")[1].lower()
    if file.mimetype not in ("video/mp4","video/webm","video/quicktime","video/x-msvideo","video/hevc"):
        return jsonify({"ok": False, "error": "unsupported video type"}), 400
    if ext not in (".mp4",".webm",".mov",".avi",".hevc"):
        return jsonify({"ok": False, "error": "unsupported video extension"}), 400
    if not file:
        return jsonify({"ok": False, "error": "empty file"}), 400
    with NamedTemporaryFile(delete=False, suffix=ext or ".mp4") as tmp_in:
        file.save(tmp_in)
        tmp_in_path = tmp_in.name
    try:
        faces = probe_video_faces(tmp_in_path, max_frames=120, fast=True)
        return jsonify({"ok": True, "faces": int(faces)})
    finally:
        try:
            os.remove(tmp_in_path)
        except:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    try:
        print(app.url_map)
    except Exception:
        pass
    app.run(host="127.0.0.1", port=port, debug=False)
