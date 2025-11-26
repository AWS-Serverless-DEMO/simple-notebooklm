"""Document text extraction module for PDF, DOCX, and TXT files"""

from typing import List, Dict
from pypdf import PdfReader
from docx import Document
import io


class DocumentProcessor:
    """Extract text from various document formats"""

    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
        """
        Extract text from PDF file with page metadata

        Args:
            file_bytes: PDF file content as bytes
            filename: Name of the PDF file

        Returns:
            List of dictionaries with text and metadata per page
        """
        results = []
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():  # Only include non-empty pages
                results.append({
                    'text': text,
                    'metadata': {
                        'document': filename,
                        'page': page_num,
                        'total_pages': len(reader.pages),
                        'source_type': 'pdf'
                    }
                })

        return results

    @staticmethod
    def extract_text_from_docx(file_bytes: bytes, filename: str) -> List[Dict]:
        """
        Extract text from DOCX file with metadata

        Args:
            file_bytes: DOCX file content as bytes
            filename: Name of the DOCX file

        Returns:
            List with single dictionary containing all text and metadata
        """
        docx_file = io.BytesIO(file_bytes)
        doc = Document(docx_file)

        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])

        return [{
            'text': text,
            'metadata': {
                'document': filename,
                'page': 1,
                'total_pages': 1,
                'source_type': 'docx'
            }
        }]

    @staticmethod
    def extract_text_from_txt(file_bytes: bytes, filename: str) -> List[Dict]:
        """
        Extract text from TXT file

        Args:
            file_bytes: TXT file content as bytes
            filename: Name of the TXT file

        Returns:
            List with single dictionary containing all text and metadata
        """
        text = file_bytes.decode('utf-8')

        return [{
            'text': text,
            'metadata': {
                'document': filename,
                'page': 1,
                'total_pages': 1,
                'source_type': 'txt'
            }
        }]

    @classmethod
    def process_document(cls, file_bytes: bytes, filename: str) -> List[Dict]:
        """
        Process document based on file extension

        Args:
            file_bytes: File content as bytes
            filename: Name of the file

        Returns:
            List of dictionaries with extracted text and metadata

        Raises:
            ValueError: If file type is not supported
        """
        file_extension = filename.lower().split('.')[-1]

        if file_extension == 'pdf':
            return cls.extract_text_from_pdf(file_bytes, filename)
        elif file_extension in ['docx', 'doc']:
            return cls.extract_text_from_docx(file_bytes, filename)
        elif file_extension == 'txt':
            return cls.extract_text_from_txt(file_bytes, filename)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
