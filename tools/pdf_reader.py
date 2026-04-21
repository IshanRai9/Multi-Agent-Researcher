import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a given PDF using PyMuPDF.
    """
    if not os.path.exists(pdf_path):
        return f"[Error] PDF file not found at {pdf_path}"
        
    try:
        doc = fitz.open(pdf_path)
        text_content = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content.append(page.get_text())
        doc.close()
        
        full_text = "\n".join(text_content).strip()
        if not full_text:
            return "[Warning] PDF is readable but contains no extractable text."
            
        return full_text
    except Exception as e:
        return f"[Error] Failed to read PDF: {str(e)}"

if __name__ == "__main__":
    test_path = os.path.join(os.path.dirname(__file__), "..", "data", "dummy.pdf")
    print(extract_text_from_pdf(test_path)[:200])
