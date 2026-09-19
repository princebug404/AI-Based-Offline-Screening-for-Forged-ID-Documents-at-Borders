# SIH26188 - Database repository (data access layer)

import json
from datetime import datetime, timezone
from database.db import get_connection


class DocumentRepository:
    """
    Data access layer for document records.

    All queries use parameterized statements to prevent SQL injection.
    Database operations are separated from Flask route logic.
    """

    def __init__(self, db_path):
        """
        Initialize the repository with a database path.

        Args:
            db_path: Absolute path to the SQLite database file.
        """
        self.db_path = db_path

    def save_document(self, document):
        """
        Insert a new document record.

        Args:
            document: Dictionary with keys:
                - id (str): UUID document ID
                - original_filename (str): Original uploaded filename
                - stored_filename (str): Safe server-side filename
                - file_type (str): File extension (e.g., 'png')
                - file_size_bytes (int): File size in bytes
                - upload_timestamp (str): ISO 8601 timestamp
                - processing_status (str): Current status (default: 'uploaded')
        """
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO documents
                    (id, original_filename, stored_filename, file_type,
                     file_size_bytes, upload_timestamp, processing_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document['id'],
                    document['original_filename'],
                    document['stored_filename'],
                    document['file_type'],
                    document['file_size_bytes'],
                    document['upload_timestamp'],
                    document.get('processing_status', 'uploaded'),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_document(self, document_id):
        """
        Retrieve a document record by ID.

        Args:
            document_id: UUID string of the document.

        Returns:
            Dictionary of document metadata, or None if not found.
        """
        conn = get_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def update_status(self, document_id, new_status):
        """
        Update the processing status of a document.

        Args:
            document_id: UUID string of the document.
            new_status: New processing status string.

        Returns:
            True if a row was updated, False if document not found.
        """
        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute(
                "UPDATE documents SET processing_status = ? WHERE id = ?",
                (new_status, document_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def claim_processing(self, document_id):
        """Claim a document for processing if another request has not claimed it."""
        conn = get_connection(self.db_path)
        try:
            cursor = conn.execute(
                """
                UPDATE documents
                SET processing_status = 'processing'
                WHERE id = ? AND processing_status != 'processing'
                """,
                (document_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_documents(self):
        """
        Retrieve all document records, ordered by upload time (newest first).

        Returns:
            List of document dictionaries.
        """
        conn = get_connection(self.db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY upload_timestamp DESC"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # --- Processing results ---

    @staticmethod
    def _redact_fields(fields):
        """Return sanitized field values suitable for persistence and API responses."""
        if isinstance(fields, str):
            try:
                fields = json.loads(fields)
            except (TypeError, ValueError):
                return {}

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

    @classmethod
    def sanitize_processing_result(cls, result):
        """Sanitize a processing record before writing it to SQLite or returning it."""
        if not result:
            return result

        safely_redacted = dict(result)
        safely_redacted['extracted_text'] = None
        safely_redacted['extracted_fields'] = json.dumps(
            cls._redact_fields(result.get('extracted_fields', {}))
        )
        safely_redacted['identity_match'] = json.dumps(
            cls._sanitize_identity_match(result.get('identity_match'))
        ) if result.get('identity_match') is not None else None
        return safely_redacted

    @staticmethod
    def _sanitize_identity_match(identity_match):
        """Keep only synthetic match metadata and field-level outcomes."""
        if not isinstance(identity_match, dict):
            return {}

        allowed_statuses = {'Match', 'Mismatch', 'Unavailable'}
        fields = identity_match.get('fields', {})
        if not isinstance(fields, dict):
            fields = {}
        safe_fields = {
            field: status
            for field, status in fields.items()
            if field in {'document_type', 'document_number', 'name', 'date_of_birth'}
            and status in allowed_statuses
        }
        return {
            'status': identity_match.get('status'),
            'record_id': identity_match.get('record_id'),
            'synthetic_database': identity_match.get('synthetic_database') is True,
            'fields': safe_fields,
        }

    def save_processing_result(self, result):
        """
        Insert or replace a processing result record.

        Uses INSERT OR REPLACE so re-processing overwrites previous results.

        Args:
            result: Dictionary with keys:
                - document_id (str): UUID of the document
                - document_type (str|None): Detected type
                - confidence (float|None): Classification confidence
                - extracted_text (str|None): Raw OCR text
                - extracted_fields (str): JSON string of extracted fields
                - ocr_engine (str|None): Engine identifier
                - ocr_confidence (float|None): OCR confidence
                - processing_error (str|None): Error message if failed
        """
        safe_result = self.sanitize_processing_result(result)

        conn = get_connection(self.db_path)
        try:
            self._insert_processing_result(conn, safe_result)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _insert_processing_result(conn, safe_result):
        conn.execute(
            """
            INSERT OR REPLACE INTO processing_results
                (document_id, document_type, confidence, extracted_text,
                 extracted_fields, ocr_engine, ocr_confidence,
                  processing_error, identity_match, processed_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                safe_result['document_id'],
                safe_result.get('document_type'),
                safe_result.get('confidence'),
                safe_result.get('extracted_text'),
                safe_result.get('extracted_fields', '{}'),
                safe_result.get('ocr_engine'),
                safe_result.get('ocr_confidence'),
                safe_result.get('processing_error'),
                safe_result.get('identity_match'),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def save_processing_result_and_status(self, result, new_status):
        """Atomically persist a processing result and its document status."""
        safe_result = self.sanitize_processing_result(result)

        conn = get_connection(self.db_path)
        try:
            self._insert_processing_result(conn, safe_result)
            cursor = conn.execute(
                "UPDATE documents SET processing_status = ? WHERE id = ?",
                (new_status, safe_result['document_id']),
            )
            if cursor.rowcount == 0:
                raise ValueError("Document not found")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_processing_result(self, document_id):
        """
        Retrieve processing results for a document.

        Args:
            document_id: UUID string of the document.

        Returns:
            Dictionary of processing results, or None if not found.
        """
        conn = get_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM processing_results WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if row is None:
                return None

            result = dict(row)
            result['extracted_text'] = None
            if result.get('extracted_fields'):
                try:
                    raw_fields = json.loads(result['extracted_fields'])
                except (TypeError, ValueError):
                    raw_fields = {}
                result['extracted_fields'] = json.dumps(self._redact_fields(raw_fields))
            else:
                result['extracted_fields'] = '{}'
            if result.get('identity_match'):
                try:
                    result['identity_match'] = json.loads(result['identity_match'])
                except (TypeError, ValueError):
                    result['identity_match'] = None
            else:
                result['identity_match'] = None
            return result
        finally:
            conn.close()


