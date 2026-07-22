from pypdf import PdfReader

def load_pdf(pdf_source):
    """
    Extracts text from a PDF file path or file-like object.
    """
    reader = PdfReader(pdf_source)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
