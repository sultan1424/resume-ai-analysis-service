import io
import logging
import pypdf

logger = logging.getLogger(__name__)


def extract_text(cv_bytes: bytes) -> str:
    """
    Extract plain text from a CV PDF file.

    Args:
        cv_bytes: raw bytes of the PDF file read from S3

    Returns:
        extracted text as a single string
    """
    try:
        # Load PDF from bytes (not from disk — Factor 6: stateless)
        pdf_reader = pypdf.PdfReader(io.BytesIO(cv_bytes))

        text_parts = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = "\n".join(text_parts).strip()

        if not full_text:
            raise ValueError("No text could be extracted from the CV")

        logger.info(f"Extracted {len(full_text)} characters from PDF")
        return full_text

    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise
