# OCR / PDF-to-Word Quality Gate

## Arabic OCR benchmark
Build a held-out corpus of at least 100 pages with verified ground-truth text:
- clean digital text, scanned books, low-resolution camera captures
- Naskh and Ruq'ah where available, mixed Arabic/English, numerals and punctuation
- multi-column pages, tables, forms, stamps and skewed pages

For every page, save the source image/PDF and UTF-8 reference text. Never tune on the held-out test set.

Run Tesseract-only, PaddleOCR-only and hybrid on the same pages. Report macro-average and per-document CER/WER, exact-page rate, empty-page rate, latency and peak memory. A candidate is accepted only when CER and WER improve without a material regression in layout/latency.

Arabic normalization used for scoring removes tashkeel/tatweel and normalizes common alef/yeh variants. Also report raw CER/WER so normalization cannot hide meaningful errors.

## PDF to DOCX benchmark
Use a fixed suite of text PDFs, image-only scans, mixed pages, tables, multi-column pages, headers/footers, RTL paragraphs and embedded images. Compare extracted text, page/paragraph order, table cell content, image count and visual rendering of DOCX-to-PDF against the source. Record failures separately; do not describe a conversion as lossless unless verified.

## Reproducibility
Store corpus version, engine versions, DPI, language, preprocessing settings and per-file results. Keep private or copyrighted documents out of public repositories; publish only licensed test data or synthetic fixtures.
