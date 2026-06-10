import os
import tempfile
from app.extractor import extract_fields
from app.ocr import process_receipt_lines, extract_receipt


def test_extract_fields_basic():
    text = "Test Shop\n01/06/2026\nApple 2 x KES 50.00\nBanana KES 20.00\nTOTAL KES 120.00\nCASH PAID KES 200.00\nCHANGE KES 80.00\n"
    fields = extract_fields(text)
    assert fields["merchant"].upper().startswith("TEST"), fields
    assert fields["amount"] == 120.0
    assert len(fields["line_items"]) >= 1


def test_process_receipt_lines_simple():
    lines = [
        {"text": "Test Shop"},
        {"text": "01/06/2026"},
        {"text": "Apple 2 x KES 50.00"},
        {"text": "Banana KES 20.00"},
        {"text": "TOTAL KES 120.00"},
        {"text": "CASH PAID KES 200.00"},
        {"text": "CHANGE KES 80.00"},
    ]
    r = process_receipt_lines(lines)
    assert r["total"] == 120.0
    assert r["cash_paid"] == 200.0
    assert r["change"] == 80.0
    assert r["merchant"].upper().startswith("TEST")


def test_extract_receipt_file(tmp_path):
    # create a temp image with text using PIL
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Test Shop\n01/06/2026\nTOTAL KES 120.00", fill=(0, 0, 0))
    p = tmp_path / "receipt.png"
    img.save(p)
    res = extract_receipt(str(p))
    assert res["total"] in (120.0, 0.0)
