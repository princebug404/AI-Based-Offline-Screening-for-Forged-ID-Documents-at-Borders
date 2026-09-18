# SIH26188 - Document processing service
#
# Orchestrates the document processing pipeline:
#   1. Load document metadata from database
#   2. Run OCR text extraction
#   3. Extract structured fields from OCR text
#   4. Classify document type
#   5. Save processing results to database
#   6. Update document status

import os
import logging
import json

from src.ocr.ocr_engine import OCREngine
from src.ocr.field_extraction import FieldExtractor
from src.vision.document_classfication import DocumentClassifier
from backend.services.identity_matching import SyntheticIdentityMatcher
from database.repository import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentProcessingService:
    """
    Orchestrates OCR, field extraction, and classification for uploaded documents.

    This service coordinates the processing pipeline and manages status
    transitions: uploaded → processing → processed / failed.
    """

    def __init__(self, db_path, upload_folder, identity_records_path):
        """
        Initialize the processing service.

        Args:
            db_path: Absolute path to the SQLite database.
            upload_folder: Absolute path to the upload directory.
        """
        self.db_path = db_path
        self.upload_folder = upload_folder
        self.ocr_engine = OCREngine()
        self.field_extractor = FieldExtractor()
        self.classifier = DocumentClassifier()
        self.identity_matcher = SyntheticIdentityMatcher(identity_records_path)
        self.repo = DocumentRepository(db_path)

    @staticmethod
    def _redact_fields(fields):
        """Return field values that are safe to share in API responses."""
        if not isinstance(fields, dict):
            return {}

        redacted = {}
        for key, value in fields.items():
            if key in {'name', 'date_of_birth', 'document_number', 'expiry_date'}:
                redacted[key] = '[REDACTED]' if value not in (None, '') else None
            elif key == 'all_dates':
                redacted[key] = []
            else:
                redacted[key] = value
        return redacted

    def _save_failed_result(self, document_id, classification, fields, ocr_result, error):
        """Persist available processing output while marking the document failed."""
        result_record = {
            "document_id": document_id,
            "document_type": classification.get("document_type", "unknown"),
            "confidence": classification.get("confidence", 0.0),
            "extracted_text": None,
            "extracted_fields": json.dumps(self._redact_fields(fields)),
            "ocr_engine": ocr_result.get("engine", "tesseract"),
            "ocr_confidence": ocr_result.get("confidence"),
            "processing_error": error,
        }
        self.repo.save_processing_result_and_status(result_record, "failed")

        return {
            "success": False,
            "error": error,
            "document_id": document_id,
            "processing_status": "failed",
            "document_type": result_record["document_type"],
            "type_confidence": result_record["confidence"],
            "classification_method": classification.get("method"),
            "extracted_text": None,
            "extracted_fields": self._redact_fields(fields),
            "ocr_engine": result_record["ocr_engine"],
            "ocr_confidence": result_record["ocr_confidence"],
        }

    def process_document(self, document_id):
        """Claim a document and run its processing pipeline once."""
        doc = self.repo.get_document(document_id)
        if doc is None:
            return {
                "success": False,
                "error": "Document not found",
                "document_id": document_id,
            }

        file_path = os.path.join(self.upload_folder, doc["stored_filename"])
        if not os.path.isfile(file_path):
            self.repo.update_status(document_id, "failed")
            return {
                "success": False,
                "error": "Uploaded file not found on disk",
                "document_id": document_id,
            }

        # The conditional database update makes this claim atomic across requests.
        if not self.repo.claim_processing(document_id):
            return {
                "success": False,
                "error": "Document is already being processed",
                "document_id": document_id,
                "processing_status": "processing",
            }

        try:
            return self._process_claimed_document(document_id, file_path)
        except Exception as error:
            logger.exception("Document processing failed")
            self.repo.update_status(document_id, "failed")
            return {
                "success": False,
                "error": f"Processing failed: {error}",
                "document_id": document_id,
                "processing_status": "failed",
            }

    def _process_claimed_document(self, document_id, file_path):
        """
        Run the full processing pipeline on an uploaded document.

        Pipeline:
            1. Look up document in database
            2. Verify the file exists on disk
            3. Update status to 'processing'
            4. Run OCR to extract text
            5. Extract structured fields from text
            6. Classify the document type
            7. Save results and update status to 'processed' or 'failed'

        Args:
            document_id: UUID of the uploaded document.

        Returns:
            Dictionary with processing results, or error information.
        """
        # --- 1. Run OCR ---
        ocr_result = self.ocr_engine.extract_text(file_path)

        if not ocr_result["success"]:
            # OCR failed (e.g., Tesseract not installed, corrupt image)
            result_record = {
                "document_id": document_id,
                "document_type": "unknown",
                "confidence": 0.0,
                "extracted_text": None,
                "extracted_fields": "{}",
                "ocr_engine": ocr_result.get("engine", "tesseract"),
                "ocr_confidence": None,
                "processing_error": ocr_result.get("error", "OCR failed"),
            }
            self.repo.save_processing_result_and_status(result_record, "failed")

            return {
                "success": False,
                "error": ocr_result["error"],
                "document_id": document_id,
                "processing_status": "failed",
            }

        extracted_text = ocr_result["text"]

        # --- 5. Extract structured fields ---
        fields = self.field_extractor.extract_fields(extracted_text)
        redacted_fields = self._redact_fields(fields)

        # --- 6. Classify document type ---
        classification = self.classifier.classify(extracted_text, image_path=file_path)

        if not extracted_text or not extracted_text.strip():
            return self._save_failed_result(
                document_id,
                classification,
                fields,
                ocr_result,
                "OCR completed but returned no readable text",
            )

        classification_confidence = classification.get("confidence", 0.0)
        if classification.get("document_type") == "unknown":
            return self._save_failed_result(
                document_id,
                classification,
                fields,
                ocr_result,
                "Document classification was unknown",
            )

        if classification_confidence < 0.15:
            return self._save_failed_result(
                document_id,
                classification,
                fields,
                ocr_result,
                "Document classification confidence is below the minimum threshold",
            )

        identity_match = self.identity_matcher.match(
            classification['document_type'],
            fields,
        )

        # --- 7. Save results and update status ---
        result_record = {
            "document_id": document_id,
            "document_type": classification["document_type"],
            "confidence": classification["confidence"],
            "extracted_text": None,
            "extracted_fields": json.dumps(redacted_fields),
            "ocr_engine": ocr_result.get("engine", "tesseract"),
            "ocr_confidence": ocr_result.get("confidence"),
            "processing_error": None,
        }
        self.repo.save_processing_result_and_status(result_record, "processed")

        return {
            "success": True,
            "document_id": document_id,
            "processing_status": "processed",
            "document_type": classification["document_type"],
            "type_confidence": classification["confidence"],
            "classification_method": classification["method"],
            "extracted_text": None,
            "extracted_fields": redacted_fields,
            "ocr_engine": ocr_result.get("engine", "tesseract"),
            "ocr_confidence": ocr_result.get("confidence"),
            "identity_match": identity_match,
        }

