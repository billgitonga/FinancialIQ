# app/ocr.py

from __future__ import annotations

import os
import re
import cv2
import json
import hashlib
import tempfile
import logging
import numpy as np
import pytesseract
from pathlib import Path
from typing import List, Dict, Optional
from pdf2image import convert_from_path

from app.extractor import extract_fields, extract_total_amount

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION (from environment variables)
# ============================================================

# Speed/quality trade-off: use a single best PSM by default
DEFAULT_PSM_MODES = os.getenv("OCR_PSM_MODES", "6,4,3")  # comma-separated, e.g., "6,4,3"
try:
    PSM_MODES = [int(m.strip()) for m in DEFAULT_PSM_MODES.split(",") if m.strip()]
except ValueError:
    PSM_MODES = [6]

# Enable EasyOCR fallback (very slow) – disabled by default
EASYOCR_FALLBACK = os.getenv("OCR_EASYOCR_FALLBACK", "false").lower() == "true"

# Upscale width (set to 0 to disable upscaling)
UPSCALE_WIDTH = int(os.getenv("OCR_UPSCALE_WIDTH", "1400"))
PDF_DPI = int(os.getenv("OCR_PDF_DPI", "200"))

# Cache OCR results (enabled by default)
USE_CACHE = os.getenv("OCR_USE_CACHE", "true").lower() == "true"
CACHE_DIR = Path(os.getenv("OCR_CACHE_DIR", "ocr_cache"))

# Other constants
MIN_CONFIDENCE = int(os.getenv("OCR_MIN_CONFIDENCE", "10"))
FALLBACK_CONFIDENCE = int(os.getenv("OCR_FALLBACK_CONFIDENCE", "45"))
ADAPTIVE_BLOCK_SIZE = int(os.getenv("OCR_ADAPTIVE_BLOCK_SIZE", "31"))
ADAPTIVE_C = int(os.getenv("OCR_ADAPTIVE_C", "12"))
MORPH_CLOSE_KERNEL = (1, 2)  # horizontal closing
GAUSSIAN_BLUR_KERNEL = (3, 3)
TESSERACT_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,/-$%& "

USE_CLAHE = os.getenv("OCR_USE_CLAHE", "true").lower() == "true"
CLAHE_CLIP_LIMIT = float(os.getenv("OCR_CLAHE_CLIP_LIMIT", "2.0"))
CLAHE_GRID_SIZE = int(os.getenv("OCR_CLAHE_GRID_SIZE", "8"))
SHARPEN_STRENGTH = float(os.getenv("OCR_SHARPEN_STRENGTH", "0.5"))
OCR_LLM_FALLBACK = os.getenv("OCR_LLM_FALLBACK", "false").lower() == "true"
OLLAMA_OCR_MODEL = os.getenv("OLLAMA_OCR_MODEL", os.getenv("OLLAMA_MODEL", "tinyllama"))

# Enhanced preprocessing options
USE_DENOISING = os.getenv("OCR_USE_DENOISING", "true").lower() == "true"
USE_CONTRAST_ENHANCEMENT = os.getenv("OCR_USE_CONTRAST_ENHANCEMENT", "true").lower() == "true"
USE_THERMAL_PAPER_MODE = os.getenv("OCR_USE_THERMAL_PAPER_MODE", "true").lower() == "true"
MAX_IMAGE_SIZE = int(os.getenv("OCR_MAX_IMAGE_SIZE", "3000"))

STOPWORDS = {"TOTAL", "VAT", "CHANGE", "CASH", "PAID", "AMOUNT", "BALANCE", "RECEIPT", "DATE", "TIME", "TEL", "PIN", "CODE", "QTY", "SUBTOTAL", "DISCOUNT", "CASHIER", "BILL", "TABLE", "TAX", "GRAND TOTAL", "CASH TENDERED", "CARD", "TERMINAL", "REF", "INVOICE", "INV NO", "ORDER", "NO.", "RECEIPT NO", "CASHIER ID", "POS", "SHOP", "STORE", "WWW", "HTTP"}
SUMMARY_KEYWORDS = ["TOTAL", "SUBTOTAL", "VAT", "CHANGE", "CASH PAID", "GRAND TOTAL", "TOTAL AMOUNT", "CASH TENDERED", "CARD PAYMENT"]
ITEM_SECTION_KEYWORDS = ["ITEM", "DESCRIPTION", "QTY", "PRICE", "AMOUNT", "PRODUCT", "SERVICE"]
REWRITE_HINTS = ["SUBTOTAL", "TOTAL", "VAT", "CHANGE", "CASH", "PAID", "BALANCE", "RECEIPT", "DATE", "TIME"]

# Lazy load EasyOCR if needed
_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if EASYOCR_FALLBACK and _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR loaded (fallback)")
        except ImportError:
            logger.warning("EasyOCR not installed. Install with: pip install easyocr")
    return _easyocr_reader


# ============================================================
# CACHING
# ============================================================

def _file_hash(file_path: str) -> str:
    """Return hash of file content + modification time."""
    stat = os.stat(file_path)
    with open(file_path, "rb") as f:
        content_hash = hashlib.md5(f.read()).hexdigest()
    return hashlib.md5(f"{content_hash}_{stat.st_mtime}".encode()).hexdigest()


def _get_cached_ocr(file_path: str) -> Optional[str]:
    if not USE_CACHE:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = _file_hash(file_path)
    cache_file = CACHE_DIR / f"{h}.txt"
    if cache_file.exists():
        logger.debug(f"OCR cache hit for {file_path}")
        return cache_file.read_text(encoding="utf-8")
    return None


def _set_cached_ocr(file_path: str, text: str) -> None:
    if not USE_CACHE:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = _file_hash(file_path)
    cache_file = CACHE_DIR / f"{h}.txt"
    cache_file.write_text(text, encoding="utf-8")


# ============================================================
# IMAGE PREPROCESSING (fast path)
# ============================================================

def preprocess_image_fast(image_path: str) -> np.ndarray:
    """Fast preprocessing: grayscale, noise reduction, adaptive threshold, optional resize."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Deskew only if angle > 1 degree (skip for speed)
    angle = _get_skew_angle(image)
    if abs(angle) > 1.0:
        image = _rotate_image(image, angle)

    # Resize if width < UPSCALE_WIDTH (improves OCR speed and accuracy)
    h, w = image.shape[:2]
    if UPSCALE_WIDTH > 0 and w < UPSCALE_WIDTH:
        scale = UPSCALE_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_GRID_SIZE, CLAHE_GRID_SIZE))
        gray = clahe.apply(gray)

    if SHARPEN_STRENGTH > 0:
        kernel = np.array([[-1, -1, -1],
                            [-1, 9 + SHARPEN_STRENGTH * 4, -1],
                            [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)

    return gray


def preprocess_image_enhanced(image_path: str) -> np.ndarray:
    """Enhanced preprocessing for better OCR accuracy on challenging receipts."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Limit size for performance
    h, w = image.shape[:2]
    if MAX_IMAGE_SIZE > 0 and max(h, w) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thermal paper mode: reduce contrast and remove background patterns
    if USE_THERMAL_PAPER_MODE:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        gray = thresh

    # Denoising
    if USE_DENOISING:
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Contrast enhancement
    if USE_CONTRAST_ENHANCEMENT and not USE_THERMAL_PAPER_MODE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT * 1.5, tileGridSize=(CLAHE_GRID_SIZE, CLAHE_GRID_SIZE))
        gray = clahe.apply(gray)

    # Sharpen
    if SHARPEN_STRENGTH > 0:
        kernel = np.array([[-1, -1, -1],
                            [-1, 9 + SHARPEN_STRENGTH * 6, -1],
                            [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)

    return gray


def _get_skew_angle(image: np.ndarray) -> float:
    """Estimate skew angle using a quick text mask and minimum area rectangle."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh < 255))
    if len(coords) < 100:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _crop_text_regions(image: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 60 or h < 30 or w * h < 1500:
            continue
        regions.append((y, x, w, h))
    regions.sort()
    return [image[y:y+h, x:x+w] for y, x, w, h in regions]


# ============================================================
# TESSERACT OCR
# ============================================================

def extract_text_tesseract(image: np.ndarray) -> tuple[str, float]:
    """Run Tesseract with configured PSM modes and return the best text and confidence."""
    best_text = ""
    best_score = -1.0
    best_confidence = 0.0
    for psm in PSM_MODES:
        config = (
            f"--oem 3 --psm {psm} "
            f"-c preserve_interword_spaces=1 "
            f"-c tessedit_char_whitelist=\"{TESSERACT_WHITELIST}\""
        )
        raw_text = pytesseract.image_to_string(image, config=config)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
        words = []
        confidences = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = 0.0
            confidences.append(conf)
            if conf < MIN_CONFIDENCE:
                continue
            words.append(text)
        result = raw_text.strip() or " ".join(words).strip()
        if result:
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            score = len(result) * (avg_conf / 100.0)
            if score > best_score:
                best_score = score
                best_text = result
                best_confidence = avg_conf
    return best_text, best_confidence


def extract_text_from_image(image_path: str, enhanced: bool = False) -> tuple[str, float]:
    """Main entry: returns OCR text and confidence (with caching and optional fallback).

    Args:
        image_path: Path to image file
        enhanced: If True, use enhanced preprocessing for better accuracy (slower)
    """
    cached = _get_cached_ocr(image_path)
    if cached is not None:
        return cached, 60.0

    if enhanced:
        processed = preprocess_image_enhanced(image_path)
    else:
        processed = preprocess_image_fast(image_path)
    text, confidence = extract_text_tesseract(processed)

    if EASYOCR_FALLBACK and (len(text.strip()) < 40 or confidence < FALLBACK_CONFIDENCE):
        logger.info("Tesseract output was low-confidence; trying EasyOCR fallback")
        reader = _get_easyocr_reader()
        if reader:
            try:
                result = reader.readtext(image_path, detail=0, paragraph=True)
                easy_text = ' '.join(result)
                if len(easy_text.strip()) > len(text.strip()):
                    text = easy_text
                    confidence = 50.0
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")

    _set_cached_ocr(image_path, text)
    return text, confidence


# ============================================================
# PDF handling (unchanged, but uses fast image processing)
# ============================================================

def pdf_to_images(pdf_path: str) -> List[str]:
    pages = convert_from_path(pdf_path, dpi=PDF_DPI)  # balanced DPI for speed and accuracy
    image_paths = []
    for idx, page in enumerate(pages):
        temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        page.save(temp.name)
        image_paths.append(temp.name)
    return image_paths


def cleanup_temp_files(paths: List[str]) -> None:
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file: {path} | {e}")


def extract_text_from_pdf(pdf_path: str) -> str:
    temp_files = pdf_to_images(pdf_path)
    all_text = []
    try:
        for image_path in temp_files:
            text, _ = extract_text_from_image(image_path)
            all_text.append(text)
    finally:
        cleanup_temp_files(temp_files)
    return "\n".join(all_text)


# ============================================================
# Receipt parsing (same as before, but uses fast OCR)
# ============================================================

def detect_merchant(lines: List[Dict]) -> str:
    top_lines = lines[:20]
    candidates = []
    for line in top_lines:
        text = line["text"].strip()
        if len(text) < 3:
            continue
        upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        score = 0
        if upper_ratio > 0.5:
            score += 3
        if line.get("y", 0) < 350:
            score += 2
        if len(text.split()) <= 6:
            score += 2
        if not re.search(r'\d', text):
            score += 1
        if any(word in text.upper() for word in SUMMARY_KEYWORDS):
            score -= 5
        candidates.append((score, text))
    candidates.sort(reverse=True)
    if candidates:
        return candidates[0][1].title()

    return "Unknown Merchant"


def extract_date(text: str) -> Optional[str]:
    patterns = [r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})', r'(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})']
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
    return None


def detect_regions(lines: List[Dict]) -> Dict:
    item_start = None
    summary_start = None
    for idx, line in enumerate(lines):
        text = line["text"].upper()
        if any(key in text for key in ITEM_SECTION_KEYWORDS):
            item_start = idx
        if any(key in text for key in SUMMARY_KEYWORDS):
            summary_start = idx
            break
    return {"item_start": item_start, "summary_start": summary_start}


def parse_money(value: str) -> Optional[float]:
    if not value:
        return None
    value = value.replace(",", "")
    value = re.sub(r"[^\d.\-]", "", value)
    try:
        return float(value)
    except Exception:
        return None


def extract_line_items(lines: List[Dict], regions: Dict) -> List[Dict]:
    items = []
    if regions["item_start"] is None:
        return items
    start = regions["item_start"] + 1
    end = regions["summary_start"] if regions["summary_start"] is not None else len(lines)
    for line in lines[start:end]:
        text = line["text"]
        amounts = re.findall(r'(\d+[.,]?\d{0,2})', text)
        if not amounts:
            continue
        total = parse_money(amounts[-1])
        if total is None or total <= 0:
            continue
        desc = text
        for amount in amounts:
            desc = desc.replace(amount, "")
        desc = re.sub(r'\b\d+\b', '', desc)
        desc = re.sub(r'[^A-Za-z0-9\s\-&]', '', desc)
        desc = re.sub(r'\s+', ' ', desc).strip()
        if not desc:
            continue
        if any(stop in desc.upper() for stop in STOPWORDS):
            continue
        items.append({"description": desc.title(), "quantity": 1, "unit_price": total, "total": total})
    return items


def extract_total(lines: List[Dict]) -> float:
    full_text = "\n".join(line["text"] for line in lines)
    total = extract_total_amount(full_text)
    return total if total is not None else 0.0


def extract_field(lines: List[Dict], keywords: List[str]) -> Optional[float]:
    for line in lines:
        text = line["text"].upper()
        if any(keyword in text for keyword in keywords):
            amounts = re.findall(r'(\d+[.,]?\d{0,2})', text)
            if amounts:
                return parse_money(amounts[-1])
    return None


def process_receipt_lines(lines: List[Dict]) -> Dict:
    full_text = "\n".join(line["text"] for line in lines)
    fields = extract_fields(full_text)
    merchant = fields.get("merchant") or detect_merchant(lines)
    receipt_date = fields.get("date") or extract_date(full_text)
    items = []
    for item in fields.get("line_items", []):
        description = item.get("description", "Item").strip() or "Item"
        if any(keyword in description.upper() for keyword in SUMMARY_KEYWORDS + list(STOPWORDS)):
            continue
        qty = int(item.get("quantity", 1)) if str(item.get("quantity", 1)).isdigit() else 1
        unit_price = item.get("unit_price")
        total_line = item.get("total") if item.get("total") is not None else (unit_price * qty if unit_price is not None else None)
        if unit_price is None and total_line is None:
            continue
        items.append({
            "description": description,
            "quantity": qty,
            "unit_price": float(unit_price) if unit_price is not None else None,
            "total": float(total_line),
        })

    total = fields.get("amount") or extract_total(lines) or 0.0
    cash_paid = fields.get("cash_paid")
    change = fields.get("change")
    vat = fields.get("vat")
    if total <= 0 and items:
        total = sum(item["total"] for item in items)
    if not items and total > 0:
        items.append({"description": f"Purchase At {merchant}", "quantity": 1, "unit_price": total, "total": total})
    return {
        "merchant": merchant,
        "date": receipt_date,
        "items": items,
        "total": total,
        "cash_paid": cash_paid,
        "change": change,
        "vat": vat,
        "raw_text": full_text,
    }


def extract_receipt(file_path: str) -> Dict:
    file_path = str(Path(file_path).resolve())
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    ext = Path(file_path).suffix.lower()
    all_lines = []
    temp_files = []
    try:
        if ext == ".pdf":
            temp_files = pdf_to_images(file_path)
            for image_path in temp_files:
                text, _ = extract_text_from_image(image_path)
                for line in text.split("\n"):
                    if line.strip():
                        all_lines.append({"text": line.strip(), "y": 0})
        else:
            text, _ = extract_text_from_image(file_path)
            for line in text.split("\n"):
                if line.strip():
                    all_lines.append({"text": line.strip(), "y": 0})
        receipt_data = process_receipt_lines(all_lines)

        parsed_conf = _calc_parsed_confidence(receipt_data, all_lines)

        if OCR_LLM_FALLBACK and parsed_conf < float(os.getenv("OCR_LLM_FALLBACK_CONFIDENCE", "35")):
            logger.info("Standard OCR confidence %.1f below threshold; attempting LLM rewrite", parsed_conf)
            llm_data = _llm_parse_receipt(receipt_data.get("raw_text", ""))
            if llm_data and any(llm_data.get(k) for k in ("merchant", "total", "items")) and len(llm_data.get("items", [])) >= len(receipt_data.get("items", [])):
                receipt_data = _merge_llm_result(receipt_data, llm_data)

        receipt_data = _validate_receipt(receipt_data)
        return receipt_data
    finally:
        cleanup_temp_files(temp_files)


def extract_full_receipt_info(file_path: str) -> Dict:
    return extract_receipt(file_path)


def match_invoice_to_receipts(invoice: Dict, receipts: List[Dict], date_tolerance_days: int = 3, amount_tolerance: float = 0.05) -> List[Dict]:
    """Attempt to match an invoice dict to existing receipt records.

    invoice: {"merchant": str, "date": "YYYY-MM-DD" or similar, "amount": float}
    receipts: list of structured receipts (as returned by extract_receipt)

    Returns list of candidate matches with a simple score.
    """
    from datetime import datetime

    def _date_to_dt(d):
        try:
            return datetime.fromisoformat(str(d))
        except Exception:
            try:
                return datetime.strptime(str(d), "%d/%m/%Y")
            except Exception:
                return None

    inv_amt = float(invoice.get("amount") or 0)
    inv_date = _date_to_dt(invoice.get("date"))
    inv_merchant = (invoice.get("merchant") or "").lower()

    candidates = []

    for r in receipts:
        r_amt = float(r.get("total") or 0)
        # amount similarity
        amt_score = 1.0 - (abs(r_amt - inv_amt) / max(inv_amt, 1.0)) if inv_amt else 0.0

        # date proximity
        r_date = _date_to_dt(r.get("date"))
        date_score = 0.0
        if inv_date and r_date:
            days = abs((inv_date - r_date).days)
            date_score = max(0.0, 1.0 - (days / max(1.0, date_tolerance_days)))

        # merchant similarity via token overlap
        r_merchant = (r.get("merchant") or "").lower()
        a = set(inv_merchant.split())
        b = set(r_merchant.split())
        merchant_score = 0.0
        if a and b:
            merchant_score = len(a & b) / max(1, len(a | b))

    # merchant fuzzy matching if rapidfuzz available
    try:
        from rapidfuzz import fuzz
        def merchant_similarity(a, b):
            try:
                return fuzz.token_set_ratio(a, b) / 100.0
            except Exception:
                return 0.0
    except Exception:
        def merchant_similarity(a, b):
            a_set = set(str(a or "").lower().split())
            b_set = set(str(b or "").lower().split())
            if not a_set or not b_set:
                return 0.0
            return len(a_set & b_set) / max(1, len(a_set | b_set))

    inv_amt = float(invoice.get("amount") or 0)
    inv_date = _date_to_dt(invoice.get("date"))
    inv_merchant = (invoice.get("merchant") or "").lower()

    candidates = []

    for r in receipts:
        r_amt = float(r.get("total") or 0)
        # amount similarity
        amt_score = 1.0 - (abs(r_amt - inv_amt) / max(inv_amt, 1.0)) if inv_amt else 0.0

        # date proximity
        r_date = _date_to_dt(r.get("date"))
        date_score = 0.0
        if inv_date and r_date:
            days = abs((inv_date - r_date).days)
            date_score = max(0.0, 1.0 - (days / max(1.0, date_tolerance_days)))

        # merchant similarity via fuzzy or token overlap
        r_merchant = (r.get("merchant") or "").lower()
        merchant_score = merchant_similarity(inv_merchant, r_merchant)

        # combined score
        score = (amt_score * 0.6) + (date_score * 0.25) + (merchant_score * 0.15)

        candidates.append({"receipt": r, "score": float(score), "amt_score": float(amt_score), "date_score": float(date_score), "merchant_score": float(merchant_score)})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    # filter by threshold
    return [c for c in candidates if c["score"] > 0.35]


# ============================================================
# LLM-based receipt understanding (optional fallback)
# ============================================================

def _calc_parsed_confidence(receipt: Dict, lines: List[Dict]) -> float:
    score = 0.0
    total = float(receipt.get("total") or 0)
    if total > 0:
        score += 35
    if receipt.get("merchant") and receipt["merchant"] != "Unknown Merchant":
        score += 25
    if receipt.get("date"):
        score += 15
    items = receipt.get("items") or []
    if items:
        score += min(25, len(items) * 5)
    raw_text = receipt.get("raw_text", "")
    if raw_text and len(raw_text.strip()) > 40:
        score += 10
    return max(0.0, min(100.0, score))


def _llm_parse_receipt(raw_text: str) -> Dict[str, Any]:
    from app.llm import ask_ollama
    if not raw_text or not raw_text.strip():
        return {}
    prompt = (
        "You are an expert receipt parser. "
        "Extract structured data from the raw receipt OCR text below. "
        "Return ONLY valid JSON with keys: merchant, date, total, cash_paid, change, vat, items. "
        "For items use list of {description, quantity, unit_price, total}. "
        "Preserve original descriptions from the receipt. If a field is missing, use null or empty string/array. "
        "If the receipt appears to be for fuel, groceries, pharmacy, restaurant, or any other business, capture that in merchant. "
        "Example:\n"
        '{ "merchant": "Shell", "date": "2025-03-12", "total": 4500.0, "cash_paid": 5000.0, "change": 500.0, "vat": 675.0, "items": [ {"description": "Unleaded 92", "quantity": 1, "unit_price": 4500.0, "total": 4500.0} ] }'
        "\n\nReceipt text:\n" + raw_text
    )
    response = ask_ollama(prompt, system="You are a receipt parser returning JSON only.", model=OLLAMA_OCR_MODEL, timeout=30)
    if not response:
        return {}
    try:
        data = json.loads(response.strip())
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        logger.warning("LLM receipt parse returned non-JSON output")
    return {}


def _merge_llm_result(base: Dict, llm: Dict) -> Dict:
    merged = dict(base)
    for key in ["merchant", "date", "total", "cash_paid", "change", "vat"]:
        val = llm.get(key)
        if val not in (None, "", 0):
            merged[key] = val
    llm_items = llm.get("items") or []
    if llm_items:
        merged["items"] = llm_items
    merged["raw_text"] = base.get("raw_text", "") or merged.get("raw_text", "")
    return merged


def _validate_receipt(receipt: Dict) -> Dict:
    total = float(receipt.get("total") or 0)
    items = receipt.get("items") or []
    if total <= 0 and items:
        total = sum(float(i.get("total") or 0) for i in items)
        receipt["total"] = total
    if not items and total > 0 and not receipt.get("merchant"):
        receipt["merchant"] = "Unknown Merchant"
    return receipt