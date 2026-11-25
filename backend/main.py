from __future__ import annotations
import io, os, re, json, shutil, tempfile, threading
from typing import Any
import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

APP_NAME = "Alfalqi PDF OCR API"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "12"))
MAX_CONVERT_PAGES = int(os.getenv("MAX_CONVERT_PAGES", "20"))
OCR_DPI = int(os.getenv("OCR_DPI", "240"))
_paddle_lock = threading.Lock()
_paddle_instances: dict[str, Any] = {}

app = FastAPI(title=APP_NAME, version="0.2.0", description="Hybrid Arabic/English OCR and PDF-to-DOCX")
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

def _check_size(data: bytes):
    if not data:
        raise HTTPException(400, "الملف فارغ")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"حجم الملف يتجاوز الحد ({MAX_UPLOAD_MB} MB)")

def _language(language: str):
    value = (language or "ara+eng").lower()
    if value in ("ar", "ara", "arabic"): return "ara", "ar"
    if value in ("en", "eng", "english"): return "eng", "en"
    return "ara+eng", "ar"

def _arabic(text: str) -> bool:
    return any("\u0600" <= c <= "\u06ff" or "\ufb50" <= c <= "\ufeff" for c in text)

def _decode(data: bytes):
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise HTTPException(415, "تعذرت قراءة الصورة؛ استخدم PNG أو JPG") from exc

def _preprocess(image):
    h, w = image.shape[:2]
    scale = min(2.0, max(1.0, 1800 / max(h, w)))
    if scale > 1.01:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _get_paddle(language):
    _, lang = _language(language)
    with _paddle_lock:
        if lang not in _paddle_instances:
            from paddleocr import PaddleOCR
            try:
                _paddle_instances[lang] = PaddleOCR(lang=lang, device="cpu",
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=True)
            except TypeError:
                _paddle_instances[lang] = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False, use_gpu=False)
        return _paddle_instances[lang]

def _paddle(image, language):
    try:
        engine = _get_paddle(language)
        result = engine.predict(image) if hasattr(engine, "predict") else engine.ocr(image, cls=True)
        rows = []
        for item in result or []:
            if hasattr(engine, "predict"):
                payload = item.json if hasattr(item, "json") else item
                if callable(payload): payload = payload()
                if isinstance(payload, str): payload = json.loads(payload)
                payload = payload.get("res", payload) if isinstance(payload, dict) else {}
                texts, scores = payload.get("rec_texts", []), payload.get("rec_scores", [])
                boxes = payload.get("rec_polys", payload.get("dt_polys", []))
                for txt, score, box in zip(texts, scores, boxes):
                    pts = np.asarray(box, dtype=float).reshape(-1, 2)
                    if len(pts) >= 4:
                        x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
                        rows.append({"text": str(txt).strip(), "confidence": float(score)*100,
                                     "bbox": [int(x0), int(y0), int(x1), int(y1)], "engine": "paddleocr"})
            else:
                for line in item or []:
                    box, pair = line[0], line[1]
                    pts = np.asarray(box, dtype=float).reshape(-1, 2)
                    x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
                    rows.append({"text": str(pair[0]).strip(), "confidence": float(pair[1])*100,
                                 "bbox": [int(x0), int(y0), int(x1), int(y1)], "engine": "paddleocr"})
        return [r for r in rows if r["text"]], None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"

def _tesseract(image, language):
    tess_lang, _ = _language(language)
    try:
        rows = []
        for psm in (6, 11):
            data = pytesseract.image_to_data(image, lang=tess_lang,
                config=f"--oem 1 --psm {psm} -c preserve_interword_spaces=1",
                output_type=pytesseract.Output.DICT)
            groups = {}
            for i, raw in enumerate(data.get("text", [])):
                word = re.sub(r"\s+", " ", raw or "").strip()
                if not word: continue
                key = tuple(data[k][i] for k in ("page_num","block_num","par_num","line_num"))
                groups.setdefault(key, []).append(i)
            for indices in groups.values():
                words = []
                for i in indices:
                    word = (data["text"][i] or "").strip()
                    try: conf = float(data["conf"][i])
                    except (ValueError, TypeError): conf = 0
                    if word and conf >= 10:
                        words.append((int(data["left"][i]), int(data["top"][i]),
                            int(data["width"][i]), int(data["height"][i]), conf, word))
                if not words: continue
                rtl = any(_arabic(w[5]) for w in words)
                words.sort(key=lambda w:w[0], reverse=rtl)
                x0=min(w[0] for w in words); y0=min(w[1] for w in words)
                x1=max(w[0]+w[2] for w in words); y1=max(w[1]+w[3] for w in words)
                rows.append({"text":" ".join(w[5] for w in words),
                    "confidence":sum(w[4] for w in words)/len(words),
                    "bbox":[x0,y0,x1,y1],"engine":f"tesseract_psm{psm}"})
        return rows, None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"

def _iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1]); x1,y1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0)
    area_a=max(1,(a[2]-a[0])*(a[3]-a[1])); area_b=max(1,(b[2]-b[0])*(b[3]-b[1]))
    return inter/max(1,min(area_a,area_b))

def _merge(rows):
    kept=[]
    for row in sorted(rows,key=lambda x:x["confidence"],reverse=True):
        if not any(_iou(row["bbox"], old["bbox"]) > .55 for old in kept): kept.append(row)
    return kept

def _order(rows):
    rows=sorted(rows,key=lambda r:((r["bbox"][1]+r["bbox"][3])/2,r["bbox"][0]))
    bands=[]
    for row in rows:
        cy=(row["bbox"][1]+row["bbox"][3])/2; h=max(1,row["bbox"][3]-row["bbox"][1])
        band=next((b for b in reversed(bands[-3:]) if abs(cy-sum((x["bbox"][1]+x["bbox"][3])/2 for x in b)/len(b)) <= .55*max(h,sum(x["bbox"][3]-x["bbox"][1] for x in b)/len(b))),None)
        if band is None:
            bands.append([row])
        else:
            band.append(row)
    out=[]
    for band in bands:
        band.sort(key=lambda r:r["bbox"][0],reverse=any(_arabic(r["text"]) for r in band))
        out.extend(band)
    return out

def _run_ocr(image, language):
    prepared=_preprocess(image)
    paddle, pe=_paddle(prepared,language)
    tess, te=_tesseract(prepared,language)
    rows=_order(_merge(paddle+tess))
    engine="hybrid" if paddle and tess else "paddleocr" if paddle else "tesseract" if tess else "none"
    h,w=prepared.shape[:2]
    return {"engine":engine,"image_width":w,"image_height":h,"lines":rows,
            "engines":{"paddleocr":{"available":pe is None,"error":pe},
                       "tesseract":{"available":te is None,"error":te}}}

@app.get("/api/health")
def health():
    import importlib.util
    return {"ok":True,"service":APP_NAME,"version":"0.2.0",
        "paddleocr_installed":importlib.util.find_spec("paddleocr") is not None,
        "pdf2docx_installed":importlib.util.find_spec("pdf2docx") is not None,
        "tesseract_languages":pytesseract.get_languages(config="") if shutil.which("tesseract") else [],
        "max_upload_mb":MAX_UPLOAD_MB,"max_pdf_pages":MAX_PDF_PAGES,
        "max_convert_pages":MAX_CONVERT_PAGES}

@app.post("/api/ocr/image")
async def ocr_image(file: UploadFile=File(...), language: str=Form("ara+eng")):
    data=await file.read(); _check_size(data)
    result=_run_ocr(_decode(data),language)
    if not result["lines"]:
        raise HTTPException(503,{"message":"لم يتمكن محركا OCR من استخراج النص","engines":result["engines"]})
    return result

@app.post("/api/ocr/pdf")
async def ocr_pdf(file: UploadFile=File(...), language: str=Form("ara+eng"), dpi: int=Form(OCR_DPI)):
    data=await file.read(); _check_size(data)
    try: doc=fitz.open(stream=data,filetype="pdf")
    except Exception as exc: raise HTTPException(415,"ملف PDF غير صالح أو محمي") from exc
    if doc.needs_pass: doc.close(); raise HTTPException(423,"ملف PDF محمي بكلمة مرور")
    if doc.page_count>MAX_PDF_PAGES: doc.close(); raise HTTPException(413,"عدد الصفحات يتجاوز الحد")
    scale=max(1,min(4,dpi/72)); pages=[]
    try:
        for n,page in enumerate(doc):
            pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
            arr=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)
            image=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
            pages.append({"page":n+1,"ocr":_run_ocr(image,language)})
    finally: doc.close()
    return {"page_count":len(pages),"pages":pages}

@app.post("/api/convert/pdf-to-docx")
async def pdf_to_docx(file: UploadFile=File(...)):
    data=await file.read(); _check_size(data)
    try: doc=fitz.open(stream=data,filetype="pdf")
    except Exception as exc: raise HTTPException(415,"ملف PDF غير صالح") from exc
    if doc.needs_pass: doc.close(); raise HTTPException(423,"أزل حماية PDF قبل التحويل")
    count=doc.page_count; doc.close()
    if count>MAX_CONVERT_PAGES: raise HTTPException(413,"عدد الصفحات يتجاوز الحد")
    folder=tempfile.mkdtemp(prefix="alfalqi-docx-")
    src=os.path.join(folder,"source.pdf"); out=os.path.join(folder,"converted.docx")
    with open(src,"wb") as f: f.write(data)
    converter=None
    try:
        from pdf2docx import Converter
        converter=Converter(src); converter.convert(out,start=0,end=count)
        if not os.path.isfile(out) or os.path.getsize(out)<100:
            raise RuntimeError("ملف DOCX الناتج فارغ أو غير صالح")
        return FileResponse(out,media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename="alfalqi-converted.docx",background=BackgroundTask(shutil.rmtree,folder,ignore_errors=True))
    except ImportError as exc:
        shutil.rmtree(folder,ignore_errors=True)
        raise HTTPException(503,"محرك pdf2docx غير مثبت") from exc
    except HTTPException: raise
    except Exception as exc:
        shutil.rmtree(folder,ignore_errors=True)
        raise HTTPException(422,f"تعذر التحويل: {str(exc)[:200]}") from exc
    finally:
        if converter is not None:
            try: converter.close()
            except Exception: pass
