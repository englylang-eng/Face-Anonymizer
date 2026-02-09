import os
from io import BytesIO
import numpy as np
import cv2
from flask import Flask, request, send_file, send_from_directory, jsonify
from tempfile import NamedTemporaryFile
from anonymizer import process_array, process_video
import mimetypes
import shutil
from flask_cors import CORS

app = Flask(__name__, static_folder="web")
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/images/<path:filename>")
def images(filename):
    root = os.path.join(os.path.dirname(__file__), "images")
    return send_from_directory(root, filename)

@app.get("/images")
def images_list():
    root = os.path.join(os.path.dirname(__file__), "images")
    try:
        files = os.listdir(root)
    except Exception:
        files = []
    return jsonify({"files": files})
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
    if file.mimetype not in ("image/jpeg","image/png","image/webp"):
        return jsonify({"error": "unsupported image type"}), 400
    if ext not in (".jpg",".jpeg",".png",".webp"):
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

@app.post("/api/anonymize_video")
def anonymize_video():
    if "video" not in request.files:
        return jsonify({"error": "no video"}), 400
    file = request.files["video"]
    ext = os.path.splitext(file.filename or "")[1].lower()
    if file.mimetype not in ("video/mp4","video/webm"):
        return jsonify({"error": "unsupported video type"}), 400
    if ext not in (".mp4",".webm"):
        return jsonify({"error": "unsupported video extension"}), 400
    if not file:
        return jsonify({"error": "empty file"}), 400
    method = request.form.get("type", "blur")
    try:
        intensity = int(request.form.get("intensity", "30"))
    except:
        intensity = 30
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        file.save(tmp_in)
        tmp_in_path = tmp_in.name
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
        out_path = tmp_out.name
    try:
        output_path, total_faces = process_video(tmp_in_path, out_path, method=method, intensity=intensity)
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    try:
        print(app.url_map)
    except Exception:
        pass
    app.run(host="127.0.0.1", port=port, debug=False)
