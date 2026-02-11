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

def count_faces_array(img) -> int:
    face_cascade = load_cascade()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return int(len(faces))

def _normalize_method(method: str) -> str:
    m = (method or "blur").lower().strip()
    if m in ("remap_features","feature","features","landmarks","facial_features"):
        return "pixelate"
    if m in ("pixelate","pixelation","mosaic","tiles"):
        return "pixelate"
    if m in ("bar","blackbar","black_bar","censor","censored_bar"):
        return "black_bar"
    if m in ("gaussian","gaussian_blur","blur"):
        return "blur"
    return m

 

def _apply_anonymization(img, rects: List[Tuple[int,int,int,int]], method: str = "blur", intensity: int = 30):
    method = _normalize_method(method)
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

# Landmark-based code removed
def process_array(img, method: str = "blur", intensity: int = 30):
    face_cascade = load_cascade()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    rects = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    method = _normalize_method(method)
    img = _apply_anonymization(img, rects, method=method, intensity=intensity)
    return img, len(rects)

 

def _ffmpeg_path() -> str:
    return shutil.which("ffmpeg") or ""

def _transcode_to_h264(src: str, dst: str) -> bool:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i", src,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        dst,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except Exception:
        return False
    return proc.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0

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

def process_video(input_path: str, output_path: str, method: str = "blur", intensity: int = 30, fast: bool = False, progress_cb=None) -> Tuple[str, int]:
    if not os.path.exists(input_path):
        print(f"Error: Video file not found at {input_path}")
        return "", 0
    read_path = input_path
    cap = cv2.VideoCapture(read_path)
    if not cap.isOpened():
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        trans_path = os.path.join(out_dir, "_transcoded_h264.mp4")
        if _transcode_to_h264(input_path, trans_path):
            read_path = trans_path
            cap = cv2.VideoCapture(read_path)
    if not cap.isOpened():
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
    method = _normalize_method(method)
    target_detect_width = 480 if fast else 640
    detect_every = 3 if fast else 2
    frame_index = 0
    last_rects: List[Tuple[int,int,int,int]] = []
    prev_gray = None
    points_list: List[np.ndarray] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
            points_list = []
            if len(last_rects) > 0:
                for (x, y, w, h) in last_rects:
                    x0 = max(0, x)
                    y0 = max(0, y)
                    x1 = min(width, x + w)
                    y1 = min(height, y + h)
                    sub = gray[y0:y1, x0:x1]
                    pts = cv2.goodFeaturesToTrack(sub, maxCorners=25 if not fast else 15, qualityLevel=0.01, minDistance=5)
                    if pts is not None and len(pts) > 0:
                        pts = pts.reshape(-1, 2)
                        pts[:, 0] += x0
                        pts[:, 1] += y0
                        points_list.append(pts.astype(np.float32))
                    else:
                        points_list.append(np.empty((0, 2), dtype=np.float32))
        else:
            rects = last_rects
            if prev_gray is not None and len(points_list) == len(last_rects) and len(last_rects) > 0:
                updated_rects: List[Tuple[int,int,int,int]] = []
                new_points_list: List[np.ndarray] = []
                for i, (x, y, w, h) in enumerate(last_rects):
                    pts_prev = points_list[i]
                    if pts_prev is None or len(pts_prev) == 0:
                        updated_rects.append((x, y, w, h))
                        new_points_list.append(pts_prev)
                        continue
                    pts_prev_klt = pts_prev.reshape(-1, 1, 2)
                    pts_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev_klt, None, winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
                    if pts_next is None or st is None:
                        updated_rects.append((x, y, w, h))
                        new_points_list.append(pts_prev)
                        continue
                    st = st.reshape(-1)
                    valid_prev = pts_prev[st == 1]
                    valid_next = pts_next.reshape(-1, 2)[st == 1]
                    if len(valid_prev) >= 4 and len(valid_next) >= 4:
                        dxy = valid_next - valid_prev
                        dx = np.median(dxy[:, 0])
                        dy = np.median(dxy[:, 1])
                        nx = int(round(x + dx))
                        ny = int(round(y + dy))
                        nx = max(0, min(nx, width - 1))
                        ny = max(0, min(ny, height - 1))
                        updated_rects.append((nx, ny, w, h))
                        new_points_list.append(valid_next.astype(np.float32))
                    else:
                        updated_rects.append((x, y, w, h))
                        new_points_list.append(valid_next.astype(np.float32) if valid_next is not None else np.empty((0, 2), dtype=np.float32))
                rects = updated_rects
                last_rects = rects
                points_list = []
                for i, (x, y, w, h) in enumerate(rects):
                    pts = new_points_list[i] if i < len(new_points_list) else np.empty((0, 2), dtype=np.float32)
                    if pts is None or len(pts) < 6:
                        x0 = max(0, x)
                        y0 = max(0, y)
                        x1 = min(width, x + w)
                        y1 = min(height, y + h)
                        sub = gray[y0:y1, x0:x1]
                        rep = cv2.goodFeaturesToTrack(sub, maxCorners=25 if not fast else 15, qualityLevel=0.01, minDistance=5)
                        if rep is not None and len(rep) > 0:
                            rep = rep.reshape(-1, 2)
                            rep[:, 0] += x0
                            rep[:, 1] += y0
                            pts = rep.astype(np.float32)
                    points_list.append(pts if pts is not None else np.empty((0, 2), dtype=np.float32))
        total_faces += len(rects)
        frame = _apply_anonymization(frame, rects, method=method, intensity=intensity)
        writer.write(frame)
        frame_index += 1
        if progress_cb and total_frames > 0:
            try:
                pct = int(min(99, max(0, (frame_index * 100) // total_frames)))
                progress_cb(pct)
            except Exception:
                pass
        prev_gray = gray
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
    if read_path != input_path:
        try:
            os.remove(read_path)
        except Exception:
            pass
    if not produced:
        print("Error: Failed to produce output video.")
        return "", total_faces
    if progress_cb:
        try:
            progress_cb(100)
        except Exception:
            pass
    return final, total_faces

def probe_video_faces(input_path: str, max_frames: int = 90, fast: bool = True) -> int:
    if not os.path.exists(input_path):
        return 0
    read_path = input_path
    cap = cv2.VideoCapture(read_path)
    if not cap.isOpened():
        out_dir = os.path.dirname(input_path) or "."
        trans_path = os.path.join(out_dir, "_probe_transcoded_h264.mp4")
        if _transcode_to_h264(input_path, trans_path):
            read_path = trans_path
            cap = cv2.VideoCapture(read_path)
    if not cap.isOpened():
        return 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    face_cascade = load_cascade()
    target_detect_width = 480 if fast else 640
    scale = 1.0
    if width > target_detect_width:
        scale = float(target_detect_width) / float(width)
    total = 0
    frames = 0
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect_gray = gray if scale >= 1.0 else cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        faces = face_cascade.detectMultiScale(
            detect_gray,
            scaleFactor=1.1 if not fast else 1.2,
            minNeighbors=5 if not fast else 4,
            minSize=(int(30 * scale), int(30 * scale)),
        )
        total += len(faces)
        frames += 1
        if total > 0:
            break
    cap.release()
    if read_path != input_path:
        try:
            os.remove(read_path)
        except Exception:
            pass
    return total
