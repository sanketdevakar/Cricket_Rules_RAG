import fitz  # PyMuPDF
import os

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input PDF path
PDF_PATH = os.path.join(PROJECT_ROOT, "data/Cricket_laws.pdf")

# Output text path
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/extracted/combined.txt")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = text.replace("\r", "\n")
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        full_text.append(text)

    return "\n\n".join(full_text)

def main():
    text = extract_text_from_pdf(PDF_PATH)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] Extracted text saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
