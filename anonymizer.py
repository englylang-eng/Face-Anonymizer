"""
Face anonymizer utilities.

Backward-compatible: public functions and signatures are preserved:
- load_cascades, load_eye_cascades, process_array, process_video, probe_video_faces, count_faces_array
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

cv2.setUseOptimized(True)

# Configure a module-level logger. Consumers can reconfigure as needed.
logger = logging.getLogger(__name__)
if not logger.handlers:
    # default handler when module executed standalone
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def _path(p: str) -> str:
    """Normalize filesystem path for OpenCV cascade loader."""
    return str(Path(p).expanduser().resolve())


@lru_cache(maxsize=2)
def load_cascades(fast: bool = False) -> List[cv2.CascadeClassifier]:
    """Load face cascade classifiers. Raises RuntimeError if none can be loaded."""
    out: List[cv2.CascadeClassifier] = []
    base = getattr(cv2.data, "haarcascades", "")
    if not base:
        raise RuntimeError("OpenCV haarcascade data directory not found (cv2.data.haarcascades).")
    names = ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"]
    if not fast:
        names += ["haarcascade_profileface.xml"]
    for n in names:
        cp = _path(Path(base) / n)
        c = cv2.CascadeClassifier(cp)
        if not c.empty():
            out.append(c)
        else:
            logger.debug("Failed to load cascade: %s", cp)
    lbp_dir = getattr(cv2.data, "lbpcascades", None)
    if lbp_dir and not fast:
        cp = _path(Path(lbp_dir) / "lbpcascade_frontalface.xml")
        c = cv2.CascadeClassifier(cp)
        if not c.empty():
            out.append(c)
    if not out:
        # Try default and raise a clear error if still not available
        cp = _path(Path(base) / "haarcascade_frontalface_default.xml")
        c = cv2.CascadeClassifier(cp)
        if c.empty():
            raise RuntimeError(f"Could not load any face cascade classifiers from {base}")
        out.append(c)
    logger.debug("Loaded %d face cascades (fast=%s)", len(out), fast)
    return out


@lru_cache(maxsize=2)
def load_eye_cascades() -> List[cv2.CascadeClassifier]:
    """Load eye cascade classifiers (cached)."""
    out: List[cv2.CascadeClassifier] = []
    base = getattr(cv2.data, "haarcascades", "")
    for n in ["haarcascade_eye.xml", "haarcascade_eye_tree_eyeglasses.xml"]:
        cp = _path(Path(base) / n)
        c = cv2.CascadeClassifier(cp)
        if not c.empty():
            out.append(c)
    logger.debug("Loaded %d eye cascades", len(out))
    return out


def _nms(rects: List[Tuple[int, int, int, int]], thr: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """Non-maximum suppression on (x,y,w,h) rectangles."""
    if not rects:
        return []
    boxes = np.array(rects, dtype=np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    keep: List[Tuple[int, int, int, int]] = []
    while order.size > 0:
        i = order[0]
        keep.append((int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])))
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


def _detect_with_cascades(
    gray: np.ndarray, cascades: List[cv2.CascadeClassifier], sf: float, mn: int, min_size_px: int
) -> List[Tuple[int, int, int, int]]:
    rects: List[Tuple[int, int, int, int]] = []
    for c in cascades:
        # detectMultiScale may return empty tuple or array
        r = c.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=(min_size_px, min_size_px))
        if r is not None and len(r) > 0:
            for (x, y, w, h) in r:
                rects.append((int(x), int(y), int(w), int(h)))
    return _nms(rects, 0.35)


def _prepare_gray(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (gray, equalized_gray) used by detectors."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ge = clahe.apply(g)
    return g, ge


def _filter_human_faces(gray: np.ndarray, rects: List[Tuple[int, int, int, int]], fast: bool) -> List[Tuple[int, int, int, int]]:
    """Remove false positive faces by checking for eyes. If no eye cascades available, returns rects unchanged."""
    eyes_c = load_eye_cascades()
    if not eyes_c:
        return rects
    out: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in rects:
        if w < 28 or h < 28:
            out.append((x, y, w, h))
            continue
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(gray.shape[1], x + w)
        y1 = min(gray.shape[0], y + h)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        # only search eyes in top ~65% of the face box
        eh = int(h * 0.65)
        ey1 = y0
        ey2 = min(y0 + eh, y1)
        eroi = gray[ey1:ey2, x0:x1]
        if eroi.size == 0:
            continue
        eyes: List[Tuple[int, int, int, int]] = []
        for ec in eyes_c:
            e = ec.detectMultiScale(eroi, scaleFactor=1.1, minNeighbors=4, minSize=(max(10, int(w * 0.12)), max(10, int(h * 0.12))))
            if e is not None and len(e) > 0:
                for (ex, ey, ew, eh2) in e:
                    eyes.append((ex, ey, ew, eh2))
        if len(eyes) >= (1 if fast else 2):
            eyes = _nms(eyes, 0.3)
            centers = [(ex + ew * 0.5, ey + eh2 * 0.5) for (ex, ey, ew, eh2) in eyes]
            ok = False
            if len(centers) >= 2:
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dx = abs(centers[i][0] - centers[j][0])
                        dy = abs(centers[i][1] - centers[j][1])
                        if dx >= w * 0.2 and dx <= w * 0.8 and dy <= h * 0.35:
                            ok = True
                            break
                    if ok:
                        break
            else:
                ok = True if fast else False
            if ok:
                out.append((x, y, w, h))
        else:
            # reject false positives
            pass
    return out


def _detect_faces_image(img: np.ndarray, strict: bool = False) -> List[Tuple[int, int, int, int]]:
    """Detect faces in a single image using multiple variants (CLAHE, unsharp)."""
    g, ge = _prepare_gray(img)
    h, w = g.shape[:2]
    minp = max(12, int(min(h, w) * 0.028))
    casc = load_cascades(fast=False)
    variants = [g, ge]
    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    us = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
    variants.append(us)
    rects: List[Tuple[int, int, int, int]] = []
    for v in variants:
        r1 = _detect_with_cascades(v, casc, sf=1.1, mn=4, min_size_px=minp)
        r2 = _detect_with_cascades(v, casc, sf=1.06, mn=3, min_size_px=minp)
        rects.extend(r1)
        rects.extend(r2)
        hh, ww = v.shape[:2]
        cy, cx = hh // 2, ww // 2
        for ang in (-12, -8, 8, 12):
            M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
            vr = cv2.warpAffine(v, M, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rr1 = _detect_with_cascades(vr, cascades=casc, sf=1.1, mn=4, min_size_px=minp)
            rr2 = _detect_with_cascades(vr, cascades=casc, sf=1.06, mn=3, min_size_px=minp)
            rects.extend(rr1)
            rects.extend(rr2)
        vf = cv2.flip(v, 1)
        rf1 = _detect_with_cascades(vf, casc, sf=1.1, mn=4, min_size_px=minp)
        rf2 = _detect_with_cascades(vf, casc, sf=1.06, mn=3, min_size_px=minp)
        rects.extend(rf1)
        rects.extend(rf2)
    rects = _nms(rects, 0.25)
    rects = _filter_human_faces(g, rects, fast=not strict)
    return rects


def count_faces_array(img: np.ndarray) -> int:
    rects = _detect_faces_image(img, strict=True)
    return int(len(rects))


def _normalize_method(method: Optional[str]) -> str:
    m = (method or "blur").lower().strip()
    if m in ("remap_features", "feature", "features", "landmarks", "facial_features"):
        return "pixelate"
    if m in ("pixelate", "pixelation", "mosaic", "tiles"):
        return "pixelate"
    if m in ("bar", "blackbar", "black_bar", "censor", "censored_bar"):
        return "black_bar"
    if m in ("gaussian", "gaussian_blur", "blur"):
        return "blur"
    return m


def _apply_anonymization(img: np.ndarray, rects: List[Tuple[int, int, int, int]], method: str = "blur", intensity: int = 30) -> np.ndarray:
    """
    Apply anonymization inplace to `img` for list of rectangles.
    - pixelate: fast resized pixelation
    - black_bar: fill with black
    - blur: Gaussian blur with kernel depending on intensity
    """
    method = _normalize_method(method)
    intensity = max(5, min(100, int(intensity)))
    # avoid blur_divisor < 1 which creates overly large kernels; clamp to 1.0 min
    blur_divisor = max(1.0, 10.0 - (intensity / 100.0) * 8.0)
    # pixel_scale used for resize-based pixelation
    pixel_scale = max(1, 2 + int((intensity / 100.0) * 18))
    for (x, y, w, h) in rects:
        max_dim = max(w, h)
        pad = int((intensity / 100.0) * 0.15 * max_dim)
        px = max(0, x - pad)
        py = max(0, y - pad)
        ex = min(img.shape[1], x + w + pad)
        ey = min(img.shape[0], y + h + pad)
        if px >= ex or py >= ey:
            continue
        roi = img[py:ey, px:ex]
        if roi.size == 0:
            continue
        if method == "pixelate":
            # faster and simpler: resize down then up
            rw = max(1, ex - px)
            rh = max(1, ey - py)
            ds_w = max(1, rw // pixel_scale)
            ds_h = max(1, rh // pixel_scale)
            small = cv2.resize(roi, (ds_w, ds_h), interpolation=cv2.INTER_LINEAR)
            # nearest gives blocky pixelation; linear can be used as well
            pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
            img[py:ey, px:ex] = pixelated
        elif method == "black_bar":
            img[py:ey, px:ex] = (0, 0, 0)
        else:
            rw = ex - px
            rh = ey - py
            kW = max(3, int(rw / blur_divisor))
            kH = max(3, int(rh / blur_divisor))
            if kW % 2 == 0:
                kW += 1
            if kH % 2 == 0:
                kH += 1
            blurred = cv2.GaussianBlur(roi, (kW, kH), 0)
            img[py:ey, px:ex] = blurred
    return img


# Landmark-based approach intentionally omitted (keeps API compatibility)


def process_array(img: np.ndarray, method: str = "blur", intensity: int = 30) -> Tuple[np.ndarray, int]:
    rects = _detect_faces_image(img, strict=False)
    img = _apply_anonymization(img, rects, method=method, intensity=intensity)
    return img, len(rects)


@lru_cache(maxsize=1)
def _ffmpeg_path() -> str:
    """Return path to ffmpeg executable or empty string if not found (cached)."""
    p = shutil.which("ffmpeg") or ""
    if not p:
        logger.debug("ffmpeg not found in PATH")
    return p


def _transcode_to_h264(src: str, dst: str) -> bool:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        src,
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        dst,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except Exception as ex:
        logger.debug("ffmpeg transcode exception: %s", ex)
        return False
    ok = proc.returncode == 0 and Path(dst).exists() and Path(dst).stat().st_size > 0
    logger.debug("transcode_to_h264(%s)->%s ok=%s rc=%s", src, dst, ok, proc.returncode)
    return ok


def _transcode_video_only(src: str, dst: str) -> bool:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        src,
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        dst,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except Exception as ex:
        logger.debug("ffmpeg transcode exception: %s", ex)
        return False
    ok = proc.returncode == 0 and Path(dst).exists() and Path(dst).stat().st_size > 0
    logger.debug("transcode_video_only(%s)->%s ok=%s rc=%s", src, dst, ok, proc.returncode)
    return ok


def _mux_audio_with_ffmpeg(video_no_audio: str, source_with_audio: str, final_out: str) -> bool:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        video_no_audio,
        "-i",
        source_with_audio,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        final_out,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except Exception as ex:
        logger.debug("ffmpeg mux exception: %s", ex)
        return False
    ok = proc.returncode == 0 and Path(final_out).exists() and Path(final_out).stat().st_size > 0
    logger.debug("mux_audio_with_ffmpeg ok=%s rc=%s", ok, proc.returncode)
    return ok


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
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


def _smooth_rects(prev: List[Tuple[int, int, int, int]], curr: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
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


def process_video(
    input_path: str,
    output_path: str,
    method: str = "blur",
    intensity: int = 30,
    fast: bool = False,
    progress_cb=None,
) -> Tuple[str, int]:
    """
    Process a video file and anonymize detected faces.

    Returns (final_output_path, total_faces_processed).
    """
    input_p = Path(input_path)
    if not input_p.exists():
        logger.error("Video file not found: %s", input_path)
        return "", 0
    read_path = str(input_p)
    cap = cv2.VideoCapture(read_path)
    if not cap.isOpened():
        out_dir = Path(output_path).parent or Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        trans_path = str(out_dir / "_transcoded_h264.mp4")
        if _transcode_to_h264(str(input_p), trans_path):
            read_path = trans_path
            cap = cv2.VideoCapture(read_path)
    if not cap.isOpened():
        logger.error("Cannot open input video (even after attempting transcode): %s", input_path)
        return "", 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = Path(output_path).parent or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_out = str(out_dir / "_temp_no_audio.mp4")
    writer = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))
    if not writer.isOpened():
        # fallback directly to requested output
        writer = cv2.VideoWriter(str(out_dir / Path(output_path).name), fourcc, fps, (width, height))
    cascades = load_cascades(fast=fast)
    total_faces = 0
    method = _normalize_method(method)
    target_detect_width = 480 if fast else 640
    detect_every = 3 if fast else 2
    frame_index = 0
    last_rects: List[Tuple[int, int, int, int]] = []
    prev_gray = None
    points_list: List[np.ndarray] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    miss_detects = 0
    persist_misses_max = int(max(6, min(24, round((fps or 25.0) * (0.6 if fast else 1.0)))))
    # local helper
    def _inflate_rects(rects: List[Tuple[int, int, int, int]], w_img: int, h_img: int, pct: float) -> List[Tuple[int, int, int, int]]:
        out_rects: List[Tuple[int, int, int, int]] = []
        for (x, y, w_r, h_r) in rects:
            cx = x + w_r * 0.5
            cy = y + h_r * 0.5
            w2 = int(round(w_r * (1.0 + pct)))
            h2 = int(round(h_r * (1.0 + pct)))
            x2 = int(round(cx - w2 * 0.5))
            y2 = int(round(cy - h2 * 0.5))
            x2 = max(0, x2)
            y2 = max(0, y2)
            if x2 + w2 > w_img:
                w2 = w_img - x2
            if y2 + h2 > h_img:
                h2 = h_img - y2
            out_rects.append((x2, y2, max(1, w2), max(1, h2)))
        return out_rects

    try:
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
                rects: List[Tuple[int, int, int, int]] = []
                inv = (1.0 / scale) if scale > 0 else 1.0
                if use_roi:
                    mx = min([x for (x, y, w, h) in last_rects])
                    my = min([y for (x, y, w, h) in last_rects])
                    mx2 = max([x + w for (x, y, w, h) in last_rects])
                    my2 = max([y + h for (x, y, w, h) in last_rects])
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
                    if roi.size == 0:
                        faces = []
                    else:
                        roi_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
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
                    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(detect_gray)
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
                if len(rects) > 0:
                    if len(last_rects) > 0:
                        rects = _smooth_rects(last_rects, rects)
                    last_rects = rects
                    miss_detects = 0
                    points_list = []
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
                    if len(last_rects) > 0 and miss_detects < persist_misses_max:
                        miss_detects += 1
                        inc = 0.03 * miss_detects
                        base = 0.10 if fast else 0.06
                        pct = min(0.25, base + inc)
                        rects = _inflate_rects(last_rects, width, height, pct)
                        detect_every = 1
                    else:
                        last_rects = []
                        rects = []
                        points_list = []
            else:
                rects = last_rects
                if prev_gray is not None and len(points_list) == len(last_rects) and len(last_rects) > 0:
                    updated_rects: List[Tuple[int, int, int, int]] = []
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
                            dxm = np.median(dxy[:, 0])
                            dym = np.median(dxy[:, 1])
                            dev = np.sqrt((dxy[:, 0] - dxm) ** 2 + (dxy[:, 1] - dym) ** 2)
                            mad = np.median(np.abs(dev - np.median(dev))) if dev.size > 0 else 0.0
                            if mad > 0:
                                msk = dev <= (3.0 * mad)
                                if np.any(msk):
                                    dxyf = dxy[msk]
                                    dxm = np.median(dxyf[:, 0])
                                    dym = np.median(dxyf[:, 1])
                            nx = int(round(x + dxm))
                            ny = int(round(y + dym))
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
            if len(rects) > 0:
                frame = _apply_anonymization(frame, rects, method=method, intensity=intensity)
            writer.write(frame)
            total_faces += len(rects)
            prev_gray = gray
            frame_index += 1
            if progress_cb is not None and total_frames > 0:
                try:
                    pct = int(min(100, max(0, (frame_index / float(total_frames)) * 100.0)))
                    progress_cb(pct)
                except Exception:
                    pass
        cap.release()
        writer.release()
        final_out = str(out_dir / Path(output_path).name)
        temp_exists = Path(temp_out).exists() and Path(temp_out).stat().st_size > 0
        if temp_exists:
            ok_mux = _mux_audio_with_ffmpeg(temp_out, read_path, final_out)
            if not ok_mux:
                ok_trans = _transcode_video_only(temp_out, final_out)
                if not ok_trans:
                    try:
                        shutil.move(temp_out, final_out)
                    except Exception:
                        final_out = temp_out
            try:
                if Path(temp_out).exists():
                    Path(temp_out).unlink()
            except Exception:
                pass
        else:
            if not Path(final_out).exists():
                final_out = str(out_dir / Path(output_path).name)
        return final_out, total_faces
    except KeyboardInterrupt:
        raise
    except Exception:
        try:
            cap.release()
            writer.release()
        except Exception:
            pass
        final_out = str(out_dir / Path(output_path).name)
        return final_out, total_faces
