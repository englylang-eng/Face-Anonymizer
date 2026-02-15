import cv2
import sys
import os
import shutil
import subprocess
from typing import Tuple, List
import numpy as np

cv2.setUseOptimized(True)

def _path(p):
    return os.path.normpath(p)

def load_cascades(fast: bool = False) -> List[cv2.CascadeClassifier]:
    out: List[cv2.CascadeClassifier] = []
    base = cv2.data.haarcascades
    names = ['haarcascade_frontalface_default.xml', 'haarcascade_frontalface_alt2.xml']
    if not fast:
        names += ['haarcascade_profileface.xml']
    for n in names:
        cp = _path(base + n)
        c = cv2.CascadeClassifier(cp)
        if not c.empty():
            out.append(c)
    lbp_dir = getattr(cv2.data, 'lbpcascades', None)
    if lbp_dir and not fast:
        cp = _path(lbp_dir + 'lbpcascade_frontalface.xml')
        c = cv2.CascadeClassifier(cp)
        if not c.empty():
            out.append(c)
    if not out:
        cp = _path(base + 'haarcascade_frontalface_default.xml')
        c = cv2.CascadeClassifier(cp)
        if c.empty():
            print("Error: Could not load face cascade classifier.")
            sys.exit(1)
        out.append(c)
    return out

def load_eye_cascades() -> List[cv2.CascadeClassifier]:
    out: List[cv2.CascadeClassifier] = []
    base = cv2.data.haarcascades
    for n in ['haarcascade_eye.xml', 'haarcascade_eye_tree_eyeglasses.xml']:
        cp = _path(base + n)
        c = cv2.CascadeClassifier(cp)
        if not c.empty():
            out.append(c)
    return out

def _nms(rects: List[Tuple[int,int,int,int]], thr: float = 0.3) -> List[Tuple[int,int,int,int]]:
    if not rects:
        return []
    boxes = np.array(rects, dtype=np.float32)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append((int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]
    return keep

def _detect_with_cascades(gray, cascades: List[cv2.CascadeClassifier], sf: float, mn: int, min_size_px: int) -> List[Tuple[int,int,int,int]]:
    rects: List[Tuple[int,int,int,int]] = []
    for c in cascades:
        r = c.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=(min_size_px, min_size_px))
        if r is not None and len(r) > 0:
            for (x,y,w,h) in r:
                rects.append((int(x), int(y), int(w), int(h)))
    return _nms(rects, 0.35)

def _prepare_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ge = clahe.apply(g)
    return g, ge

def _filter_human_faces(gray, rects: List[Tuple[int,int,int,int]], fast: bool) -> List[Tuple[int,int,int,int]]:
    eyes_c = load_eye_cascades()
    if not eyes_c:
        return rects
    out: List[Tuple[int,int,int,int]] = []
    for (x,y,w,h) in rects:
        if w < 28 or h < 28:
            out.append((x,y,w,h))
            continue
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(gray.shape[1], x + w)
        y1 = min(gray.shape[0], y + h)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        eh = int(h * 0.65)
        ey1 = y0
        ey2 = min(y0 + eh, y1)
        eroi = gray[ey1:ey2, x0:x1]
        if eroi.size == 0:
            continue
        eyes: List[Tuple[int,int,int,int]] = []
        for ec in eyes_c:
            e = ec.detectMultiScale(eroi, scaleFactor=1.1, minNeighbors=4, minSize=(max(10, int(w*0.12)), max(10, int(h*0.12))))
            if e is not None and len(e) > 0:
                for (ex,ey,ew,eh2) in e:
                    eyes.append((ex,ey,ew,eh2))
        if len(eyes) >= (1 if fast else 2):
            eyes = _nms(eyes, 0.3)
            centers = []
            for (ex,ey,ew,eh2) in eyes:
                cx = ex + ew*0.5
                cy = ey + eh2*0.5
                centers.append((cx, cy))
            ok = False
            if len(centers) >= 2:
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        dx = abs(centers[i][0] - centers[j][0])
                        dy = abs(centers[i][1] - centers[j][1])
                        if dx >= w*0.2 and dx <= w*0.8 and dy <= h*0.35:
                            ok = True
                            break
                    if ok:
                        break
            else:
                ok = True if fast else False
            if ok:
                out.append((x,y,w,h))
        else:
            pass
    return out

def count_faces_array(img) -> int:
    g, ge = _prepare_gray(img)
    minp = max(16, int(min(g.shape[0], g.shape[1]) * 0.035))
    casc = load_cascades(fast=False)
    r1 = _detect_with_cascades(g, casc, sf=1.1, mn=4, min_size_px=minp)
    r2 = _detect_with_cascades(ge, casc, sf=1.06, mn=3, min_size_px=minp)
    rects = _nms(r1 + r2, 0.35)
    rects = _filter_human_faces(g, rects, fast=False)
    return int(len(rects))

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
        max_dim = max(w, h)
        pad = int((intensity / 100.0) * 0.15 * max_dim)
        px = max(0, x - pad)
        py = max(0, y - pad)
        ex = min(img.shape[1], x + w + pad)
        ey = min(img.shape[0], y + h + pad)
        if px >= ex or py >= ey:
            continue
        if method == "pixelate":
            min_dim = min(ex - px, ey - py)
            block = int(max(8, (intensity / 100.0) * min_dim * 0.5))
            block = min(block, max(ex - px, ey - py))
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
            img[py:ey, px:ex] = (0, 0, 0)
        else:
            rw = ex - px
            rh = ey - py
            kW = max(3, int(rw / max(1.0, blur_divisor)))
            kH = max(3, int(rh / max(1.0, blur_divisor)))
            if kW % 2 == 0: kW += 1
            if kH % 2 == 0: kH += 1
            roi = img[py:ey, px:ex]
            if roi.size == 0:
                continue
            blurred = cv2.GaussianBlur(roi, (kW, kH), 0)
            img[py:ey, px:ex] = blurred
    return img

# Landmark-based code removed
def process_array(img, method: str = "blur", intensity: int = 30):
    g, ge = _prepare_gray(img)
    minp = max(16, int(min(g.shape[0], g.shape[1]) * 0.035))
    casc = load_cascades(fast=False)
    r1 = _detect_with_cascades(g, casc, sf=1.1, mn=4, min_size_px=minp)
    r2 = _detect_with_cascades(ge, casc, sf=1.06, mn=3, min_size_px=minp)
    rects = _nms(r1 + r2, 0.35)
    rects = _filter_human_faces(g, rects, fast=False)
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
        "-profile:v", "baseline",
        "-level", "3.0",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        dst,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except Exception:
        return False
    return proc.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
def _transcode_video_only(src: str, dst: str) -> bool:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i", src,
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
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
        "-profile:v", "baseline",
        "-level", "3.0",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        final_out,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except Exception:
        return False
    return proc.returncode == 0 and os.path.exists(final_out) and os.path.getsize(final_out) > 0

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1 = a[0]
    ay1 = a[1]
    ax2 = a[0] + a[2]
    ay2 = a[1] + a[3]
    bx1 = b[0]
    by1 = b[1]
    bx2 = b[0] + b[2]
    by2 = b[1] + b[3]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def _smooth_rects(prev: List[Tuple[int,int,int,int]], curr: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
    out: List[Tuple[int,int,int,int]] = []
    used = [False] * len(prev)
    for c in curr:
        best_i = -1
        best_iou = 0.0
        for i, p in enumerate(prev):
            if used[i]:
                continue
            iou = _iou(p, c)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i >= 0 and best_iou >= 0.3:
            p = prev[best_i]
            used[best_i] = True
            a = 0.5
            nx = int(round(a * p[0] + (1 - a) * c[0]))
            ny = int(round(a * p[1] + (1 - a) * c[1]))
            nw = int(round(a * p[2] + (1 - a) * c[2]))
            nh = int(round(a * p[3] + (1 - a) * c[3]))
            out.append((nx, ny, nw, nh))
        else:
            out.append(c)
    return out

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
    cascades = load_cascades(fast=fast)
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
        dyn_fast = fast and len(last_rects) > 0
        if frame_index % (detect_every if dyn_fast else 1) == 0:
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
                roi_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(roi)
                minp = max(16, int(30 * scale))
                r1 = _detect_with_cascades(roi, cascades, sf=1.12 if dyn_fast else 1.1, mn=4 if dyn_fast else 4, min_size_px=minp)
                r2 = _detect_with_cascades(roi_eq, cascades, sf=1.1 if dyn_fast else 1.06, mn=4 if dyn_fast else 3, min_size_px=minp)
                faces = _nms(r1 + r2, 0.35)
                faces = _filter_human_faces(roi, faces, fast=dyn_fast)
                for (x, y, w, h) in faces:
                    rx = int(round((x + spx) * inv))
                    ry = int(round((y + spy) * inv))
                    rw = int(round(w * inv))
                    rh = int(round(h * inv))
                    rects.append((rx, ry, rw, rh))
            else:
                eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(detect_gray)
                minp = max(16, int(30 * scale))
                r1 = _detect_with_cascades(detect_gray, cascades, sf=1.12 if dyn_fast else 1.1, mn=4 if dyn_fast else 4, min_size_px=minp)
                r2 = _detect_with_cascades(eq, cascades, sf=1.1 if dyn_fast else 1.06, mn=4 if dyn_fast else 3, min_size_px=minp)
                faces = _nms(r1 + r2, 0.35)
                faces = _filter_human_faces(detect_gray, faces, fast=dyn_fast)
                for (x, y, w, h) in faces:
                    rx = int(round(x * inv))
                    ry = int(round(y * inv))
                    rw = int(round(w * inv))
                    rh = int(round(h * inv))
                    rects.append((rx, ry, rw, rh))
            if len(last_rects) > 0 and len(rects) > 0:
                rects = _smooth_rects(last_rects, rects)
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
        elif _transcode_video_only(temp_out, final):
            produced = True
        else:
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
    cascades = load_cascades(fast=False)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    targets = [704, 576, 480] if not fast else [480]
    seen = 0
    processed = 0
    stride = 1
    if total_frames > 0:
        stride = max(1, total_frames // max_frames)
    found = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if seen % stride != 0:
            seen += 1
            continue
        seen += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for tdw in targets:
            sc = 1.0
            if width > tdw:
                sc = float(tdw) / float(width)
            det = gray if sc >= 1.0 else cv2.resize(gray, (int(width * sc), int(height * sc)), interpolation=cv2.INTER_AREA)
            eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(det)
            minp = max(12, int(24 * sc))
            r1 = _detect_with_cascades(det, cascades, sf=1.08, mn=3, min_size_px=minp)
            r2 = _detect_with_cascades(eq, cascades, sf=1.05, mn=3, min_size_px=minp)
            faces = _nms(r1 + r2, 0.35)
            faces = _filter_human_faces(det, faces, fast=True)
            if len(faces) > 0:
                found = len(faces)
                break
            # Rotation fallback for tilted faces (only if not found yet)
            h, w = det.shape[:2]
            cy, cx = h // 2, w // 2
            for ang in (-12, 12, -8, 8):
                M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
                det_r = cv2.warpAffine(det, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                eq_r = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(det_r)
                r1r = _detect_with_cascades(det_r, cascades, sf=1.08, mn=3, min_size_px=minp)
                r2r = _detect_with_cascades(eq_r, cascades, sf=1.05, mn=3, min_size_px=minp)
                fr = _nms(r1r + r2r, 0.35)
                fr = _filter_human_faces(det_r, fr, fast=True)
                if len(fr) > 0:
                    found = len(fr)
                    break
            if found > 0:
                break
        processed += 1
        if found > 0 or processed >= max_frames:
            break
    cap.release()
    if read_path != input_path:
        try:
            os.remove(read_path)
        except Exception:
            pass
    return int(found)
