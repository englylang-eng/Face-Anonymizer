# Face Anonymizer

Privacy-first image and video anonymization using OpenCV and a simple Flask web UI.

## Setup
```bash
py -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## Run
```bash
.\.venv\Scripts\python.exe web_app.py
```

Open http://127.0.0.1:5000/ and:
- Upload an image or video
- Choose anonymization method (blur, pixelate, black bar)
- Adjust intensity
- Download the anonymized result

## Notes
- All processing happens locally
- Large/unsupported codecs may fail with a clear error toast

## Requirements
- Python 3.x
- OpenCV
- NumPy
- Flask
