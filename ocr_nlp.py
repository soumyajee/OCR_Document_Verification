import pytesseract
from pdf2image import convert_from_path
import re
import json

def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text

def extract_resume_info(text):
    data = {}
    data["document_title"] = "Resume"
    # Candidate name (first line assumption)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    data["candidate_name"] = lines[0] if lines else "Unknown"

    # Email & phone
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.search(r"\+?\d[\d -]{8,12}\d", text)
    data["email"] = email.group(0) if email else "N/A"
    data["phone"] = phone.group(0) if phone else "N/A"

    # Simple extraction (extend with NLP for production)
    data["skills"] = re.findall(r"\b(Python|Java|ML|NLP|AI|SQL|TensorFlow)\b", text, re.I)
    return json.dumps(data, indent=2)
