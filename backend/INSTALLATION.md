# Runtime installation

The production container installs PaddlePaddle CPU 3.2.0 from the official Paddle CPU wheel index, then installs PaddleOCR 3.x and the Arabic/English Tesseract language packs. The API lazily initializes PaddleOCR on the first OCR request, so container startup does not download models or block the health endpoint.

`pdf2docx` is installed without its optional GUI OpenCV dependency because this service uses `opencv-python-headless`. Its runtime dependencies (Fire, FontTools, python-docx and Termcolor) are installed explicitly.

Build locally:
```bash
docker build -t alfalqi-pdf-ocr ./backend
docker run --rm -p 8000:8000 alfalqi-pdf-ocr
```

Verify installation at `GET /api/health`; it reports whether PaddleOCR and pdf2docx are installed and which Tesseract languages are present. First PaddleOCR use downloads the Arabic recognition/detection models. Ensure the deployment has outbound access and enough disk/RAM for model caching.
