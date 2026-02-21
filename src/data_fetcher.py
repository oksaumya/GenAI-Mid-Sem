import pdfplumber
from docx import Document
import wikipedia

def extract_text_from_pdf(file):
    """Extract text from an uploaded PDF file object."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        pass
    return text

def extract_text_from_docx(file):
    """Extract text from an uploaded DOCX file object."""
    text = ""
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        pass
    return text

def extract_text_from_txt(file):
    """Extract text from a TXT file object."""
    try:
        return file.getvalue().decode("utf-8")
    except Exception:
        return ""

def fetch_wikipedia_content(query):
    """Fetch Wikipedia page content for a given research query."""
    try:
        # Fetch the full page content for a better NLP corpus
        # We try to auto-suggest to get the closest match
        page = wikipedia.page(query, auto_suggest=True)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        # If ambiguous, take the first option
        try:
            return wikipedia.page(e.options[0], auto_suggest=False).content
        except:
            return ""
    except wikipedia.exceptions.PageError:
        return ""
    except Exception as e:
        return ""
