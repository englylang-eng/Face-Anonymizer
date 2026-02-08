import os
from io import BytesIO
import numpy as np
import cv2
from flask import Flask, request, send_file, send_from_directory, jsonify
from tempfile import NamedTemporaryFile
from anonymizer import process_array, process_video

app = Flask(__name__, static_folder="web")

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/api/anonymize")
def anonymize():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400
    file = request.files["image"]
    data = file.read()
    if not data:
        return jsonify({"error": "empty file"}), 400
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
    app.run(host="127.0.0.1", port=port, debug=False)
