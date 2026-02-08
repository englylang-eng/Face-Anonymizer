import cv2
import sys
import os
from typing import Tuple, List

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
            down_w = max(1, w // max(2, pixel_scale))
            down_h = max(1, h // max(2, pixel_scale))
            down = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_NEAREST)
            up = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
            img[y:y+h, x:x+w] = up
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

 

def process_video(input_path: str, output_path: str, method: str = "blur", intensity: int = 30) -> Tuple[str, int]:
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
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    face_cascade = load_cascade()
    total_faces = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        rects = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        total_faces += len(rects)
        frame = _apply_anonymization(frame, rects, method=method, intensity=intensity)
        writer.write(frame)
    cap.release()
    writer.release()
    return output_path, total_faces
