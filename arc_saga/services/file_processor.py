"""
File processing service
Handles file uploads, storage, and text extraction
"""

from pathlib import Path
from typing import Tuple, Optional
from fastapi import UploadFile
import fitz  # PyMuPDF
from docx import Document


class FileProcessor:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def process_file(
        self,
        file_id: str,
        file: UploadFile
    ) -> Tuple[str, Optional[str]]:
        """
        Save file and extract text content
        Returns: (filepath, extracted_text)
        """

        # Save file
        file_ext = Path(file.filename).suffix
        filepath = self.storage_dir / f"{file_id}{file_ext}"

        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract text based on file type
        extracted_text = None

        if file_ext == ".pdf":
            extracted_text = self._extract_pdf_text(filepath)
        elif file_ext in [".docx", ".doc"]:
            extracted_text = self._extract_docx_text(filepath)
        elif file_ext in [".txt", ".md", ".py", ".js", ".ts", ".json"]:
            extracted_text = self._extract_plain_text(filepath)

        return str(filepath), extracted_text

    def _extract_pdf_text(self, filepath: Path) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""

    def _extract_docx_text(self, filepath: Path) -> str:
        """Extract text from DOCX"""
        try:
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            print(f"DOCX extraction error: {e}")
            return ""

    def _extract_plain_text(self, filepath: Path) -> str:
        """Extract plain text files"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Text extraction error: {e}")
            return ""
