"""
Face anonymizer utilities.

Backward-compatible: public functions and signatures are preserved:
- load_cascades, load_eye_cascades, process_array, process_video, probe_video_faces, count_faces_array

Enhancements (opt-in, defaults preserve previous behavior):
 - Optional ONNX detector support (models/scrfd_2.5g.onnx)
 - Detector selection: auto|cascade|dnn|onnx (--detector)
 - Tilt/flip retry for images and selective video frames (--tilt-angles)
 - Soft-NMS option (--nms soft) and threshold (--nms-thr)
 - Adaptive min face size with optional override (--min-size-px)
 - Dynamic scaling for HD video inputs (720 px target when not fast)
 - Per-detector usage counters and end-of-run summary
 - DNN rate-limited logging and configurable cadence (--dnn-log-interval)
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

cv2.setUseOptimized(True)

# Module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Model locations (optional)
_DNN_PROTO_CANDIDATES = [
    Path(__file__).parent / "models" / "deploy.prototxt",
    Path(__file__).parent / "models" / "deploy.prototxt.txt",
]
_DNN_WEIGHTS = Path(__file__).parent / "models" / "res10_300x300_ssd_iter_140000.caffemodel"
_ONNX_MODEL = Path(__file__).parent / "models" / "scrfd_2.5g.onnx"

# Defaults / runtime config (can be set via CLI wiring below)
_DETECTOR: str = "auto"  # auto|cascade|dnn|onnx
_NMS_METHOD: str = "hard"  # hard|soft
_NMS_THR: float = 0.35
_TILT_ANGLES: List[int] = [-15, -7, 7, 15]
_MIN_SIZE_PX_OVERRIDE: Optional[int] = None
_DNN_FIRST: bool = False
_last_dnn_log_time: float = 0.0
_dnn_log_interval: float = 5.0
_SOFT_NMS_SIGMA: float = 0.5


def set_detector(det: str) -> None:
    global _DETECTOR
    _DETECTOR = (det or "auto").lower().strip()


def set_nms_method(m: str) -> None:
    global _NMS_METHOD
    _NMS_METHOD = (m or "hard").lower().strip()


def set_nms_thr(t: float) -> None:
    global _NMS_THR
    try:
        _NMS_THR = float(t)
    except Exception:
        pass


def set_min_size_px(v: Optional[int]) -> None:
    global _MIN_SIZE_PX_OVERRIDE
    if v is None:
        _MIN_SIZE_PX_OVERRIDE = None
        return
    try:
        _MIN_SIZE_PX_OVERRIDE = max(1, int(v))
    except Exception:
        pass


def set_tilt_angles_from_str(s: Optional[str]) -> None:
    global _TILT_ANGLES
    if not s:
        return
    try:
        parts = [int(x.strip()) for x in s.split(",") if x.strip()]
        if parts:
            _TILT_ANGLES = parts
    except Exception:
        pass


def set_dnn_first(v: bool) -> None:
    global _DNN_FIRST
    _DNN_FIRST = bool(v)


def set_dnn_log_interval(seconds: float) -> None:
    global _dnn_log_interval
    try:
        v = float(seconds)
        _dnn_log_interval = max(0.0, v)
    except Exception:
        pass


def _path(p: str) -> str:
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


@lru_cache(maxsize=1)
def _get_clahe():
    """Return cached CLAHE object to avoid repeated construction."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


@lru_cache(maxsize=1)
def _load_dnn_net() -> Optional[cv2.dnn_Net]:
    """Load DNN face detector if model files present. Returns None if not available.

    This supports both deploy.prototxt and deploy.prototxt.txt filenames.
    """
    try:
        proto = None
        for p in _DNN_PROTO_CANDIDATES:
            if p.exists():
                proto = p
                break
        if proto is not None and _DNN_WEIGHTS.exists():
            net = cv2.dnn.readNetFromCaffe(str(proto), str(_DNN_WEIGHTS))
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception:
                pass
            logger.info("Loaded DNN detector from %s and %s", proto, _DNN_WEIGHTS)
            return net
    except Exception as ex:
        logger.debug("Failed to load Caffe DNN net: %s", ex)
    return None


@lru_cache(maxsize=1)
def _load_onnx_net() -> Optional[cv2.dnn_Net]:
    """Load ONNX face detector if model file present. Returns None if not available.
    """
    try:
        if _ONNX_MODEL.exists():
            net = cv2.dnn.readNetFromONNX(str(_ONNX_MODEL))
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception:
                pass
            logger.info("Loaded ONNX detector from %s", _ONNX_MODEL)
            return net
        else:
            logger.debug("ONNX model not found (%s).", _ONNX_MODEL)
    except Exception as ex:
        logger.debug("Failed to load ONNX net: %s", ex)
    return None


def _detect_with_dnn(bgr: np.ndarray, net: cv2.dnn_Net, conf_thresh: float = 0.5) -> List[Tuple[int, int, int, int, float]]:
    """Use OpenCV DNN res10 SSD to detect faces. Returns list of (x,y,w,h) in image coords."""
    if net is None:
        return []
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    rects: List[Tuple[int, int, int, int, float]] = []
    for i in range(out.shape[2]):
        conf = float(out[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        x1 = int(out[0, 0, i, 3] * w)
        y1 = int(out[0, 0, i, 4] * h)
        x2 = int(out[0, 0, i, 5] * w)
        y2 = int(out[0, 0, i, 6] * h)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            rects.append((x1, y1, x2 - x1, y2 - y1, conf))
    return rects


def _detect_with_onnx(bgr: np.ndarray, net: cv2.dnn_Net, conf_thresh: float = 0.5) -> List[Tuple[int, int, int, int, float]]:
    """Basic parser for SCRFD-like outputs. Best-effort; if model outputs differ, may return []."""
    if net is None:
        return []
    h, w = bgr.shape[:2]
    # SCRFD commonly uses 640x640 normalization; try that
    blob = cv2.dnn.blobFromImage(bgr, 1.0 / 255.0, (640, 640), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    rects: List[Tuple[int, int, int, int, float]] = []
    try:
        arr = np.array(out)
        # many scrfd variants produce Nx? arrays with boxes+score. Try common layouts.
        if arr.ndim == 3:
            arr2 = arr.reshape(-1, arr.shape[-1])
            for row in arr2:
                if row.size >= 5:
                    score = float(row[-1])
                    if score < conf_thresh:
                        continue
                    x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                    # treat normalized vs absolute
                    if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0:
                        x1 = int(round(x1 * w))
                        y1 = int(round(y1 * h))
                        x2 = int(round(x2 * w))
                        y2 = int(round(y2 * h))
                    else:
                        x1 = int(round(x1)); y1 = int(round(y1)); x2 = int(round(x2)); y2 = int(round(y2))
                    if x2 > x1 and y2 > y1:
                        rects.append((max(0, x1), max(0, y1), max(1, x2 - x1), max(1, y2 - y1), score))
        elif arr.ndim == 2:
            for row in arr:
                if row.size >= 5:
                    score = float(row[4])
                    if score < conf_thresh:
                        continue
                    # interpret as cx, cy, w, h or x1,y1,x2,y2; attempt both heuristics
                    x0, y0, w0, h0 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    if 0.0 <= x0 <= 1.0 and 0.0 <= w0 <= 1.0:
                        # normalized cx, cy, w, h
                        cx = int(round(x0 * w)); cy = int(round(y0 * h))
                        rw = int(round(w0 * w)); rh = int(round(h0 * h))
                        x1 = cx - rw // 2; y1 = cy - rh // 2
                        rects.append((max(0, x1), max(0, y1), max(1, rw), max(1, rh), score))
                    else:
                        # absolute coords
                        x1 = int(round(x0)); y1 = int(round(y0)); rw = int(round(w0)); rh = int(round(h0))
                        rects.append((max(0, x1), max(0, y1), max(1, rw), max(1, rh), score))
    except Exception:
        logger.debug("Failed to parse ONNX model output", exc_info=True)
    return rects


def _nms(rects: List[Tuple[int, int, int, int]], thr: float = 0.3) -> List[Tuple[int, int, int, int]]:
    if not rects:
        return []
    boxes = np.array(rects, dtype=np.float32)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]; y2 = boxes[:, 1] + boxes[:, 3]
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


def _soft_nms(boxes_scores: List[Tuple[int, int, int, int, float]], thr: float = 0.35, sigma: float = 0.5) -> List[Tuple[int, int, int, int]]:
    if not boxes_scores:
        return []
    arr = np.array(boxes_scores, dtype=np.float32)
    x1 = arr[:, 0]; y1 = arr[:, 1]
    x2 = arr[:, 0] + arr[:, 2]; y2 = arr[:, 1] + arr[:, 3]
    scores = arr[:, 4].astype(np.float32)
    keep_idx: List[int] = []
    idxs = list(range(len(scores)))
    while idxs:
        # pick highest score index
        best_pos = int(np.argmax(scores[idxs]))
        best = idxs[best_pos]
        keep_idx.append(best)
        new_idxs = []
        for j in idxs:
            if j == best:
                continue
            xx1 = max(x1[best], x1[j]); yy1 = max(y1[best], y1[j])
            xx2 = min(x2[best], x2[j]); yy2 = min(y2[best], y2[j])
            w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
            inter = w * h
            area_i = max(0.0, (x2[best] - x1[best]) * (y2[best] - y1[best]))
            area_j = max(0.0, (x2[j] - x1[j]) * (y2[j] - y1[j]))
            union = area_i + area_j - inter
            iou = 0.0 if union <= 0 else inter / union
            scores[j] = scores[j] * math.exp(-(iou * iou) / sigma)
            if scores[j] >= 1e-3:
                new_idxs.append(j)
        idxs = new_idxs
    kept = [ (int(arr[i,0]), int(arr[i,1]), int(arr[i,2]), int(arr[i,3])) for i in keep_idx ]
    return kept


def _apply_nms_from_scores(rects_scores: List[Tuple[int, int, int, int, float]], thr: float) -> List[Tuple[int, int, int, int]]:
    if not rects_scores:
        return []
    if _NMS_METHOD == "soft":
        return _soft_nms(rects_scores, thr, sigma=_SOFT_NMS_SIGMA)
    # hard nms: discard scores
    return _nms([ (x,y,w,h) for (x,y,w,h,s) in rects_scores ], thr)


def _min_size_px_for_width(w: int) -> int:
    if _MIN_SIZE_PX_OVERRIDE is not None:
        return int(_MIN_SIZE_PX_OVERRIDE)
    w = max(1, int(w))
    if w <= 640:
        return 30
    if w >= 1920:
        return 16
    # linear interpolate between 640->30 and 1920->16
    return max(12, int(round(30 - (w - 640) * (14.0 / (1920 - 640)))))


def _prepare_gray(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = _get_clahe()
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
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(gray.shape[1], x + w); y1 = min(gray.shape[0], y + h)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        eh = int(h * 0.65)
        ey1 = y0; ey2 = min(y0 + eh, y1)
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
    return out


def _detect_faces_image(img: np.ndarray, strict: bool = False) -> List[Tuple[int, int, int, int]]:
    """Detect faces in a single image (supports detector selection, tilt, flip, soft-nms)."""
    g, ge = _prepare_gray(img)
    h, w = g.shape[:2]
    casc = load_cascades(fast=False)
    minp = _min_size_px_for_width(w)

    net_caffe = _load_dnn_net()
    net_onnx = _load_onnx_net()

    # Resolve detector choice
    detector = _DETECTOR
    if detector == "auto":
        if net_onnx is not None:
            detector = "onnx"
        elif net_caffe is not None:
            detector = "dnn"
        else:
            detector = "cascade"
    if detector == "onnx" and net_onnx is None:
        logger.info("ONNX requested but not available; falling back to cascades")
        detector = "cascade"
    if detector == "dnn" and net_caffe is None:
        logger.info("DNN requested but not available; falling back to cascades")
        detector = "cascade"

    def run_cascades_on(gray_img):
        r1 = _detect_with_cascades(gray_img, casc, sf=1.12, mn=4, min_size_px=minp)
        r2 = _detect_with_cascades(gray_img, casc, sf=1.06, mn=3, min_size_px=minp)
        return [(x, y, w0, h0, 0.5) for (x, y, w0, h0) in (r1 + r2)]

    # 1) Try 0 deg full-image
    rects_scores: List[Tuple[int, int, int, int, float]] = []
    if detector == "onnx":
        rects_scores = _detect_with_onnx(img, net_onnx, conf_thresh=0.5)
    elif detector == "dnn":
        rects_scores = _detect_with_dnn(img, net_caffe, conf_thresh=0.5)
    else:
        rects_scores = run_cascades_on(g)

    boxes = _apply_nms_from_scores(rects_scores, _NMS_THR)
    if boxes:
        boxes = _filter_human_faces(g, boxes, fast=not strict)
        if boxes:
            return boxes

    # 2) If none, try tilt angles (opt-in via _TILT_ANGLES) then flip; stop on first success.
    if _TILT_ANGLES and not strict:
        for ang in _TILT_ANGLES:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
            img_rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            g_rot = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)
            if detector == "onnx":
                rs = _detect_with_onnx(img_rot, net_onnx, conf_thresh=0.5)
            elif detector == "dnn":
                rs = _detect_with_dnn(img_rot, net_caffe, conf_thresh=0.5)
            else:
                rs = run_cascades_on(g_rot)
            boxes_rot = _apply_nms_from_scores(rs, _NMS_THR)
            if boxes_rot:
                # Map boxes back via inverse rotation
                M_inv = cv2.invertAffineTransform(M)
                mapped: List[Tuple[int, int, int, int]] = []
                for (x, y, ww, hh) in boxes_rot:
                    cx = x + ww * 0.5; cy = y + hh * 0.5
                    pt = cv2.transform(np.array([[[cx, cy]]], dtype=np.float32), M_inv)[0][0]
                    nx = int(round(pt[0] - ww * 0.5)); ny = int(round(pt[1] - hh * 0.5))
                    mapped.append((max(0, nx), max(0, ny), ww, hh))
                mapped = _filter_human_faces(g, mapped, fast=not strict)
                if mapped:
                    return mapped
        # try flipped image once
        img_flip = cv2.flip(img, 1)
        g_flip = cv2.cvtColor(img_flip, cv2.COLOR_BGR2GRAY)
        if detector == "onnx":
            rs = _detect_with_onnx(img_flip, net_onnx, conf_thresh=0.5)
        elif detector == "dnn":
            rs = _detect_with_dnn(img_flip, net_caffe, conf_thresh=0.5)
        else:
            rs = run_cascades_on(g_flip)
        boxes_flip = _apply_nms_from_scores(rs, _NMS_THR)
        if boxes_flip:
            mapped: List[Tuple[int, int, int, int]] = []
            for (x, y, ww, hh) in boxes_flip:
                nx = w - (x + ww)
                mapped.append((max(0, nx), y, ww, hh))
            mapped = _filter_human_faces(g, mapped, fast=not strict)
            if mapped:
                return mapped

    return []


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
    Improvements:
      - blends anonymized ROI back using a feathered elliptical alpha mask to avoid hard rectangular edges
      - safe, clamped Gaussian kernel sizes
      - stable resize-based pixelation
    """
    method = _normalize_method(method)
    intensity = max(5, min(100, int(intensity)))
    # avoid blur_divisor < 1 which creates overly large kernels; clamp to 1.0 min
    blur_divisor = max(1.0, 10.0 - (intensity / 100.0) * 8.0)
    # pixel_scale used for resize-based pixelation
    pixel_scale = max(1, 2 + int((intensity / 100.0) * 18))

    def _elliptical_alpha(h: int, w: int, inset_pct: float = 0.08, feather_frac: float = 0.25) -> np.ndarray:
        """
        Create an alpha mask (float32 [0..1]) of shape (h,w,1) where center ellipse ~1 and edges feather to 0.
        inset_pct: how much to shrink ellipse relative to ROI (0..0.5)
        feather_frac: fraction of max(h,w) used as Gaussian kernel for feathering

        Safety: for very small ROIs return full alpha (no feather) to avoid edge leakage.
        """
        min_dim = min(h, w)
        if min_dim < 32:
            return np.ones((h, w, 1), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        ax = int(round((w * (1.0 - inset_pct)) / 2.0))
        ay = int(round((h * (1.0 - inset_pct)) / 2.0))
        if ax <= 0 or ay <= 0:
            return np.ones((h, w, 1), dtype=np.float32)  # fallback -> fully anonymize
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        # feather by blurring the mask; kernel size should be odd and > 1
        k = max(3, int(round(max(h, w) * feather_frac)))
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(mask, (k, k), 0)
        alpha = (blurred.astype(np.float32) / 255.0)[..., np.newaxis]
        return alpha

    for (x, y, w, h) in rects:
        max_dim = max(w, h)
        pad = int((intensity / 100.0) * 0.15 * max_dim)
        px = max(0, x - pad); py = max(0, y - pad)
        ex = min(img.shape[1], x + w + pad); ey = min(img.shape[0], y + h + pad)
        if px >= ex or py >= ey:
            continue
        roi = img[py:ey, px:ex]
        if roi.size == 0:
            continue
        if method == "pixelate":
            rw = max(1, ex - px); rh = max(1, ey - py)
            ds_w = max(1, rw // pixel_scale); ds_h = max(1, rh // pixel_scale)
            small = cv2.resize(roi, (ds_w, ds_h), interpolation=cv2.INTER_LINEAR)
            anonym = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        elif method == "black_bar":
            img[py:ey, px:ex] = (0, 0, 0)
            continue
        else:
            rw = ex - px; rh = ey - py
            kW = max(3, int(rw / blur_divisor)); kH = max(3, int(rh / blur_divisor))
            if kW % 2 == 0: kW += 1
            if kH % 2 == 0: kH += 1
            kW = min(kW, max(3, rw | 1)); kH = min(kH, max(3, rh | 1))
            anonym = cv2.GaussianBlur(roi, (kW, kH), 0)
        h_roi, w_roi = anonym.shape[:2]
        alpha = _elliptical_alpha(h_roi, w_roi, inset_pct=0.08, feather_frac=0.25)
        roi_f = roi.astype(np.float32); anonym_f = anonym.astype(np.float32)
        blended = (anonym_f * alpha) + (roi_f * (1.0 - alpha))
        img[py:ey, px:ex] = np.clip(blended, 0, 255).astype(img.dtype)
    return img


def process_array(img: np.ndarray, method: str = "blur", intensity: int = 30) -> Tuple[np.ndarray, int]:
    rects = _detect_faces_image(img, strict=False)
    img = _apply_anonymization(img, rects, method=method, intensity=intensity)
    return img, len(rects)


def process_video(
    input_path: str,
    output_path: str,
    method: str = "blur",
    intensity: int = 30,
    fast: bool = False,
    progress_cb=None,
) -> Tuple[str, int]:
    global _last_dnn_log_time
    p = Path(input_path)
    if not p.exists():
        logger.error("Video file not found: %s", input_path)
        return "", 0
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        logger.error("Cannot open input video: %s", input_path)
        return "", 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))
    if not writer.isOpened():
        logger.error("Cannot open output writer: %s", output_path)
        return "", 0
    cascades = load_cascades(fast=fast)
    net_caffe = _load_dnn_net()
    net_onnx = _load_onnx_net()
    total_faces = 0
    method = _normalize_method(method)
    target_detect_width = 480 if fast else (720 if width >= 1280 else 640)
    detect_every = 3 if fast else 2
    frame_index = 0
    last_rects: List[Tuple[int, int, int, int]] = []
    prev_gray = None
    points_list: List[np.ndarray] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    miss_detects = 0
    persist_misses_max = int(max(6, min(24, round((fps or 25.0) * (0.6 if fast else 1.0)))))
    onnx_frames = 0
    dnn_frames = 0
    cascade_frames = 0
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
            frame_for_dnn = frame if scale >= 1.0 else cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
            dyn_fast = fast and len(last_rects) > 0
            rects: List[Tuple[int, int, int, int]] = []
            used_detector = None
            if frame_index % (detect_every if dyn_fast else 1) == 0:
                use_roi = len(last_rects) > 0
                inv = (1.0 / scale) if scale > 0 else 1.0
                if use_roi:
                    mx = min([x for (x, y, w, h) in last_rects]); my = min([y for (x, y, w, h) in last_rects])
                    mx2 = max([x + w for (x, y, w, h) in last_rects]); my2 = max([y + h for (x, y, w, h) in last_rects])
                    pad = int((0.12 if fast else 0.15) * width)
                    px = max(0, mx - pad); py = max(0, my - pad)
                    ex = min(width, mx2 + pad); ey = min(height, my2 + pad)
                    spx = int(round(px * scale)); spy = int(round(py * scale))
                    sex = int(round(ex * scale)); sey = int(round(ey * scale))
                    roi = detect_gray[spy:sey, spx:sex]
                    roi_color = frame_for_dnn[spy:sey, spx:sex]
                    if roi.size != 0:
                        detector = _DETECTOR
                        if detector == "auto":
                            if net_onnx is not None:
                                detector = "onnx"
                            elif net_caffe is not None:
                                detector = "dnn"
                            else:
                                detector = "cascade"
                        if detector == "onnx" and net_onnx is not None:
                            try:
                                rs = _detect_with_onnx(roi_color, net_onnx, conf_thresh=0.5)
                            except Exception:
                                rs = []
                            if rs:
                                for (dx, dy, dw, dh, s) in rs:
                                    rx = int(round((dx + spx) * inv)); ry = int(round((dy + spy) * inv))
                                    rw = int(round(dw * inv)); rh = int(round(dh * inv))
                                    rects.append((rx, ry, rw, rh))
                                used_detector = "onnx"
                        if not rects and detector in ("dnn", "cascade") and net_caffe is not None:
                            try:
                                rs = _detect_with_dnn(roi_color, net_caffe, conf_thresh=0.5)
                            except Exception:
                                rs = []
                            if rs:
                                for (dx, dy, dw, dh, s) in rs:
                                    rx = int(round((dx + spx) * inv)); ry = int(round((dy + spy) * inv))
                                    rw = int(round(dw * inv)); rh = int(round(dh * inv))
                                    rects.append((rx, ry, rw, rh))
                                used_detector = "dnn"
                        if not rects:
                            roi_eq = _get_clahe().apply(roi)
                            minp = _min_size_px_for_width(width)
                            r1 = _detect_with_cascades(roi, cascades, sf=1.12 if dyn_fast else 1.1, mn=4 if dyn_fast else 4, min_size_px=minp)
                            r2 = _detect_with_cascades(roi_eq, cascades, sf=1.1 if dyn_fast else 1.06, mn=4 if dyn_fast else 3, min_size_px=minp)
                            faces = _nms(r1 + r2, _NMS_THR)
                            faces = _filter_human_faces(roi, faces, fast=dyn_fast)
                            for (x, y, w, h) in faces:
                                rx = int(round((x + spx) * inv)); ry = int(round((y + spy) * inv))
                                rw = int(round(w * inv)); rh = int(round(h * inv))
                                rects.append((rx, ry, rw, rh))
                            if faces:
                                used_detector = "cascade"
                else:
                    detector = _DETECTOR
                    if detector == "auto":
                        if net_onnx is not None:
                            detector = "onnx"
                        elif net_caffe is not None:
                            detector = "dnn"
                        else:
                            detector = "cascade"
                    if detector == "onnx" and net_onnx is not None:
                        try:
                            rs = _detect_with_onnx(frame_for_dnn, net_onnx, conf_thresh=0.5)
                        except Exception:
                            rs = []
                        if rs:
                            if scale < 1.0:
                                for (dx, dy, dw, dh, s) in rs:
                                    rects.append((int(round(dx / scale)), int(round(dy / scale)), int(round(dw / scale)), int(round(dh / scale))))
                            else:
                                rects.extend([(x, y, w0, h0) for (x, y, w0, h0, s) in rs])
                            used_detector = "onnx"
                    if not rects and detector in ("dnn", "cascade") and net_caffe is not None:
                        try:
                            rs = _detect_with_dnn(frame_for_dnn, net_caffe, conf_thresh=0.5)
                        except Exception:
                            rs = []
                        if rs:
                            if scale < 1.0:
                                for (dx, dy, dw, dh, s) in rs:
                                    rects.append((int(round(dx / scale)), int(round(dy / scale)), int(round(dw / scale)), int(round(dh / scale))))
                            else:
                                rects.extend([(x, y, w0, h0) for (x, y, w0, h0, s) in rs])
                            used_detector = "dnn"
                            now = time.time()
                            if now - _last_dnn_log_time >= _dnn_log_interval:
                                logger.debug("DNN detection activated during video processing (frame_index=%d)", frame_index)
                                _last_dnn_log_time = now
                    if not rects:
                        eq = _get_clahe().apply(detect_gray)
                        minp = _min_size_px_for_width(width)
                        r1 = _detect_with_cascades(detect_gray, cascades, sf=1.12 if dyn_fast else 1.1, mn=4 if dyn_fast else 4, min_size_px=minp)
                        r2 = _detect_with_cascades(eq, cascades, sf=1.1 if dyn_fast else 1.06, mn=4 if dyn_fast else 3, min_size_px=minp)
                        faces = _nms(r1 + r2, _NMS_THR)
                        faces = _filter_human_faces(detect_gray, faces, fast=dyn_fast)
                        for (x, y, w, h) in faces:
                            rx = int(round(x * (1.0 / scale))); ry = int(round(y * (1.0 / scale)))
                            rw = int(round(w * (1.0 / scale))); rh = int(round(h * (1.0 / scale)))
                            rects.append((rx, ry, rw, rh))
                        if faces:
                            used_detector = "cascade"
                if len(rects) > 0:
                    if used_detector == "onnx":
                        onnx_frames += 1
                    elif used_detector == "dnn":
                        dnn_frames += 1
                    else:
                        cascade_frames += 1
                    last_rects = rects
                    miss_detects = 0
                    points_list = []
                    sub_pts: List[np.ndarray] = []
                    for (x, y, w, h) in last_rects:
                        x0 = max(0, x); y0 = max(0, y); x1 = min(width, x + w); y1 = min(height, y + h)
                        sub = gray[y0:y1, x0:x1]
                        pts = cv2.goodFeaturesToTrack(sub, maxCorners=25 if not fast else 15, qualityLevel=0.01, minDistance=5)
                        if pts is not None and len(pts) > 0:
                            pts = pts.reshape(-1, 2)
                            pts[:, 0] += x0; pts[:, 1] += y0
                            sub_pts.append(pts.astype(np.float32))
                        else:
                            sub_pts.append(np.empty((0, 2), dtype=np.float32))
                    points_list = sub_pts
                else:
                    if len(last_rects) > 0 and miss_detects < persist_misses_max:
                        miss_detects += 1
                        inc = 0.03 * miss_detects
                        base = 0.10 if fast else 0.06
                        pct = min(0.25, base + inc)
                        inflated: List[Tuple[int, int, int, int]] = []
                        for (x, y, w_r, h_r) in last_rects:
                            cx = x + w_r * 0.5; cy = y + h_r * 0.5
                            w2 = int(round(w_r * (1.0 + pct))); h2 = int(round(h_r * (1.0 + pct)))
                            x2 = int(round(cx - w2 * 0.5)); y2 = int(round(cy - h2 * 0.5))
                            x2 = max(0, x2); y2 = max(0, y2)
                            if x2 + w2 > width: w2 = width - x2
                            if y2 + h2 > height: h2 = height - y2
                            inflated.append((x2, y2, max(1, w2), max(1, h2)))
                        rects = inflated
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
                            dxm = np.median(dxy[:, 0]); dym = np.median(dxy[:, 1])
                            dev = np.sqrt((dxy[:, 0] - dxm) ** 2 + (dxy[:, 1] - dym) ** 2)
                            mad = np.median(np.abs(dev - np.median(dev))) if dev.size > 0 else 0.0
                            if mad > 0:
                                msk = dev <= (3.0 * mad)
                                if np.any(msk):
                                    dxyf = dxy[msk]
                                    dxm = np.median(dxyf[:, 0]); dym = np.median(dxyf[:, 1])
                            nx = int(round(x + dxm)); ny = int(round(y + dym))
                            nx = max(0, min(nx, width - 1)); ny = max(0, min(ny, height - 1))
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
        processed_frames = frame_index if frame_index > 0 else 1
        parts = []
        if onnx_frames > 0:
            parts.append(f"onnx:{onnx_frames} ({int(round(100.0 * onnx_frames / processed_frames))}%)")
        if dnn_frames > 0:
            parts.append(f"dnn:{dnn_frames} ({int(round(100.0 * dnn_frames / processed_frames))}%)")
        if cascade_frames > 0:
            parts.append(f"cascades:{cascade_frames} ({int(round(100.0 * cascade_frames / processed_frames))}%)")
        if parts:
            logger.info("Detectors used - " + ", ".join(parts))
        return str(out_p), total_faces
    except KeyboardInterrupt:
        raise
    except Exception:
        try:
            cap.release()
            writer.release()
        except Exception:
            pass
        return str(out_p), total_faces


def probe_video_faces(input_path: str, max_frames: int = 90, fast: bool = False) -> int:
    """Simpler probe path that respects detector selection and returns total faces across frames."""
    p = Path(input_path)
    if not p.exists():
        logger.error("Video file not found: %s", input_path)
        return 0
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        logger.error("Cannot open input video for probe: %s", input_path)
        return 0
    total = 0; frames = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_detect_width = 480 if fast else (720 if width >= 1280 else 640)
    scale = 1.0
    if width > target_detect_width:
        scale = float(target_detect_width) / float(width)
    cascades = load_cascades(fast=fast)
    net_caffe = _load_dnn_net(); net_onnx = _load_onnx_net()
    try:
        while frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detect_gray = gray if scale >= 1.0 else cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
            frame_for_dnn = frame if scale >= 1.0 else cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
            detector = _DETECTOR
            if detector == "auto":
                if net_onnx is not None:
                    detector = "onnx"
                elif net_caffe is not None:
                    detector = "dnn"
                else:
                    detector = "cascade"
            rects = []
            if detector == "onnx" and net_onnx is not None:
                rs = _detect_with_onnx(frame_for_dnn, net_onnx, conf_thresh=0.5)
                if rs:
                    if scale < 1.0:
                        for (dx, dy, dw, dh, s) in rs:
                            rects.append((int(round(dx / scale)), int(round(dy / scale)), int(round(dw / scale)), int(round(dh / scale))))
                    else:
                        rects.extend([(x,y,w0,h0) for (x,y,w0,h0,s) in rs])
            elif detector == "dnn" and net_caffe is not None:
                rs = _detect_with_dnn(frame_for_dnn, net_caffe, conf_thresh=0.5)
                if rs:
                    if scale < 1.0:
                        for (dx, dy, dw, dh, s) in rs:
                            rects.append((int(round(dx / scale)), int(round(dy / scale)), int(round(dw / scale)), int(round(dh / scale))))
                    else:
                        rects.extend([(x,y,w0,h0) for (x,y,w0,h0,s) in rs])
            else:
                minp = _min_size_px_for_width(width)
                r1 = _detect_with_cascades(detect_gray, cascades, sf=1.12, mn=4, min_size_px=minp)
                r2 = _detect_with_cascades(detect_gray, cascades, sf=1.1, mn=3, min_size_px=minp)
                faces = _nms(r1 + r2, _NMS_THR)
                faces = _filter_human_faces(detect_gray, faces, fast=fast)
                for (x, y, w0, h0) in faces:
                    rx = int(round(x / scale)); ry = int(round(y / scale)); rw = int(round(w0 / scale)); rh = int(round(h0 / scale))
                    rects.append((rx, ry, rw, rh))
            total += len(rects); frames += 1
    finally:
        try:
            cap.release()
        except Exception:
            pass
    return int(total)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="anonymizer.py CLI - process or probe videos")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_proc = sub.add_parser("process", help="Process a video and anonymize detected faces")
    p_proc.add_argument("input", help="Input video file")
    p_proc.add_argument("output", help="Output video file")
    p_proc.add_argument("--method", default="blur", help="Anonymization method: blur|pixelate|black_bar")
    p_proc.add_argument("--intensity", type=int, default=30, help="Anonymization intensity 5..100")
    p_proc.add_argument("--fast", action="store_true", help="Use fast mode")
    p_proc.add_argument("--dnn-first", action="store_true", help="Try DNN detector before cascades")
    p_proc.add_argument("--dnn-log-interval", type=float, default=None, help="Seconds between per-frame DNN debug logs (rate limit)")
    p_proc.add_argument("--detector", choices=["auto", "cascade", "dnn", "onnx"], default="auto", help="Select detector")
    p_proc.add_argument("--tilt-angles", default=None, help="Comma list of tilt angles to try, e.g. -15,-7,7,15")
    p_proc.add_argument("--nms", choices=["hard", "soft"], default="hard", help="NMS method")
    p_proc.add_argument("--nms-thr", type=float, default=None, help="NMS IoU threshold")
    p_proc.add_argument("--min-size-px", type=int, default=None, help="Override adaptive min-size-px")

    p_probe = sub.add_parser("probe", help="Probe a video to estimate faces present")
    p_probe.add_argument("input", help="Input video file")
    p_probe.add_argument("--max-frames", type=int, default=90, help="Max frames to probe")
    p_probe.add_argument("--fast", action="store_true", help="Use fast probe mode")
    p_probe.add_argument("--dnn-first", action="store_true", help="Try DNN detector before cascades")
    p_probe.add_argument("--dnn-log-interval", type=float, default=None, help="Seconds between per-frame DNN debug logs (rate limit)")
    p_probe.add_argument("--detector", choices=["auto", "cascade", "dnn", "onnx"], default="auto", help="Select detector for probe")
    p_probe.add_argument("--tilt-angles", default=None, help="Comma list of tilt angles to try, e.g. -15,-7,7,15")
    p_probe.add_argument("--nms", choices=["hard", "soft"], default="hard", help="NMS method")
    p_probe.add_argument("--nms-thr", type=float, default=None, help="NMS IoU threshold")
    p_probe.add_argument("--min-size-px", type=int, default=None, help="Override adaptive min-size-px")

    args = parser.parse_args()
    # wire CLI flags into module config
    if getattr(args, "dnn_first", False) or getattr(args, "dnn-first", False):
        set_dnn_first(True)
    if getattr(args, "dnn_log_interval", None) is not None:
        set_dnn_log_interval(getattr(args, "dnn_log_interval"))
    if getattr(args, "detector", None) is not None:
        set_detector(getattr(args, "detector"))
    if getattr(args, "nms", None) is not None:
        set_nms_method(getattr(args, "nms"))
    if getattr(args, "nms_thr", None) is not None:
        set_nms_thr(getattr(args, "nms_thr"))
    if getattr(args, "min_size_px", None) is not None:
        set_min_size_px(getattr(args, "min_size_px"))
    if getattr(args, "tilt_angles", None) is not None:
        set_tilt_angles_from_str(getattr(args, "tilt_angles"))

    if args.cmd == "process":
        out, faces = process_video(args.input, args.output, method=args.method, intensity=args.intensity, fast=args.fast)
        print(f"output: {out}   faces_detected_total: {faces}")
    elif args.cmd == "probe":
        found = probe_video_faces(args.input, max_frames=args.max_frames, fast=args.fast)
        print(found)
                  
