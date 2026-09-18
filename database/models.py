# SIH26188 - Database models (SQL schema definitions)

# SQL statement to create the documents table.
# Uses TEXT for the primary key (UUID) and ISO 8601 timestamps.
# processing_status tracks the document through the screening pipeline.

CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id                TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    stored_filename   TEXT NOT NULL,
    file_type         TEXT NOT NULL,
    file_size_bytes   INTEGER,
    upload_timestamp  TEXT NOT NULL,
    processing_status TEXT NOT NULL DEFAULT 'uploaded'
)
"""

# SQL statement to create the processing_results table.
# Stores OCR output, classification, and extracted fields for each document.
# One-to-one relationship with documents (document_id is unique here).

CREATE_PROCESSING_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS processing_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id      TEXT NOT NULL UNIQUE,
    document_type    TEXT,
    confidence       REAL,
    extracted_text   TEXT,
    extracted_fields TEXT,
    ocr_engine       TEXT,
    ocr_confidence   REAL,
    processing_error TEXT,
    processed_at     TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id)
)
"""
