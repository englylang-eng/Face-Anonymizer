import cv2
import sys
import os
import shutil
import subprocess
from typing import Tuple, List
import numpy as np

cv2.setUseOptimized(True)

def load_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier.")
        sys.exit(1)
    return face_cascade

 

def _apply_anonymization(img, rects: List[Tuple[int,int,int,int]], method: str = "blur", intensity: int = 30):
    method = (method or "blur").lower()
    intensity = max(5, min(100, int(intensity)))
    blur_divisor = 10 - (intensity / 100.0) * 8
    pixel_scale = 2 + int((intensity / 100.0) * 18)
    for (x, y, w, h) in rects:
        roi = img[y:y+h, x:x+w]
        if method == "pixelate":
            max_dim = max(w, h)
            min_dim = min(w, h)
            block = int(max(8, (intensity / 100.0) * min_dim * 0.5))
            block = min(block, max_dim)
            pad = int((intensity / 100.0) * 0.15 * max_dim)
            px = max(0, x - pad)
            py = max(0, y - pad)
            ex = min(img.shape[1], x + w + pad)
            ey = min(img.shape[0], y + h + pad)
            start_x = px - (px % block)
            start_y = py - (py % block)
            end_x = ex
            end_y = ey
            bx = start_x
            while bx < end_x:
                by = start_y
                while by < end_y:
                    tx0 = px if bx < px else bx
                    ty0 = py if by < py else by
                    tx1 = end_x if bx + block > end_x else bx + block
                    ty1 = end_y if by + block > end_y else by + block
                    if tx1 > tx0 and ty1 > ty0:
                        tile = img[ty0:ty1, tx0:tx1]
                        c = tile.mean(axis=(0, 1)).astype(np.uint8)
                        tile[:] = c
                    by += block
                bx += block
        elif method == "black_bar":
            img[y:y+h, x:x+w] = (0, 0, 0)
        else:
            kW = max(1, int(w / max(1.0, blur_divisor)))
            kH = max(1, int(h / max(1.0, blur_divisor)))
            if kW % 2 == 0: kW += 1
            if kH % 2 == 0: kH += 1
            blurred = cv2.GaussianBlur(roi, (kW, kH), 0)
            img[y:y+h, x:x+w] = blurred
    return img

def process_array(img, method: str = "blur", intensity: int = 30):
    face_cascade = load_cascade()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    rects = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    img = _apply_anonymization(img, rects, method=method, intensity=intensity)
    return img, len(rects)

 

def _ffmpeg_path() -> str:
    return shutil.which("ffmpeg") or ""

def _mux_audio_with_ffmpeg(video_no_audio: str, source_with_audio: str, final_out: str) -> bool:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i", video_no_audio,
        "-i", source_with_audio,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        final_out,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except Exception:
        return False
    return proc.returncode == 0 and os.path.exists(final_out) and os.path.getsize(final_out) > 0

def process_video(input_path: str, output_path: str, method: str = "blur", intensity: int = 30, fast: bool = False) -> Tuple[str, int]:
    if not os.path.exists(input_path):
        print(f"Error: Video file not found at {input_path}")
        return "", 0
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return "", 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    temp_out = os.path.join(out_dir, "_temp_no_audio.mp4")
    try:
        writer = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))
    except Exception:
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    face_cascade = load_cascade()
    total_faces = 0
    target_detect_width = 480 if fast else 640
    detect_every = 3 if fast else 2
    frame_index = 0
    last_rects: List[Tuple[int,int,int,int]] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 1.0
        if width > target_detect_width:
            scale = float(target_detect_width) / float(width)
        detect_gray = gray if scale >= 1.0 else cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        if frame_index % detect_every == 0:
            use_roi = len(last_rects) > 0
            rects = []
            inv = (1.0 / scale) if scale > 0 else 1.0
            if use_roi:
                mx = min([x for (x, y, w, h) in last_rects])
                my = min([y for (y, w, h) in [(r[1], r[2], r[3]) for r in last_rects]])
                mx2 = max([x + w for (x, y, w, h) in last_rects])
                my2 = max([y + h for (y, w, h) in [(r[1], r[2], r[3]) for r in last_rects]])
                pad = int((0.12 if fast else 0.15) * width)
                px = max(0, mx - pad)
                py = max(0, my - pad)
                ex = min(width, mx2 + pad)
                ey = min(height, my2 + pad)
                spx = int(round(px * scale))
                spy = int(round(py * scale))
                sex = int(round(ex * scale))
                sey = int(round(ey * scale))
                roi = detect_gray[spy:sey, spx:sex]
                faces = face_cascade.detectMultiScale(
                    roi,
                    scaleFactor=1.1 if not fast else 1.2,
                    minNeighbors=5 if not fast else 4,
                    minSize=(int(30 * scale), int(30 * scale)),
                )
                for (x, y, w, h) in faces:
                    rx = int(round((x + spx) * inv))
                    ry = int(round((y + spy) * inv))
                    rw = int(round(w * inv))
                    rh = int(round(h * inv))
                    rects.append((rx, ry, rw, rh))
            else:
                faces = face_cascade.detectMultiScale(
                    detect_gray,
                    scaleFactor=1.1 if not fast else 1.2,
                    minNeighbors=5 if not fast else 4,
                    minSize=(int(30 * scale), int(30 * scale)),
                )
                for (x, y, w, h) in faces:
                    rx = int(round(x * inv))
                    ry = int(round(y * inv))
                    rw = int(round(w * inv))
                    rh = int(round(h * inv))
                    rects.append((rx, ry, rw, rh))
            last_rects = rects
        else:
            rects = last_rects
        total_faces += len(rects)
        frame = _apply_anonymization(frame, rects, method=method, intensity=intensity)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()
    final = output_path
    produced = False
    if os.path.exists(temp_out) and os.path.getsize(temp_out) > 0:
        if _mux_audio_with_ffmpeg(temp_out, input_path, final):
            produced = True
        else:
            # Fallback: just move temp output to final (video-only)
            try:
                os.replace(temp_out, final)
                produced = True
            except Exception:
                produced = False
    else:
        produced = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    if not produced:
        print("Error: Failed to produce output video.")
        return "", total_faces
    return final, total_faces
