import os
import cv2

def main():
    base = r"d:\Python Projcts\lang_engly\web"
    src = os.path.join(base, "logo.png")
    out = os.path.join(base, "logo-white.png")
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Source not found or unreadable: {src}")
    if img.ndim == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        inv_b = 255 - b
        inv_g = 255 - g
        inv_r = 255 - r
        out_img = cv2.merge([inv_b, inv_g, inv_r, a])
    else:
        out_img = 255 - img
    ok = cv2.imwrite(out, out_img)
    if not ok:
        raise SystemExit("Failed to write output")
    print(f"Wrote {out} ({os.path.getsize(out)} bytes)")

if __name__ == "__main__":
    main()
