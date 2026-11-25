"""Text quality metrics for OCR evaluation (Arabic-aware normalization)."""
import re
import unicodedata

def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]

def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_arabic(reference), normalize_arabic(hypothesis)
    return edit_distance(list(ref), list(hyp)) / max(1, len(ref))

def wer(reference: str, hypothesis: str) -> float:
    ref = normalize_arabic(reference).split()
    hyp = normalize_arabic(hypothesis).split()
    return edit_distance(ref, hyp) / max(1, len(ref))
