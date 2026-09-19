# SIH26188 - Document processing routes

import os
import uuid
import logging
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from database.repository import DocumentRepository
from blockchain.audit_service import AuditService
from src.face.face_verfier import FaceVerifier

document_bp = Blueprint('documents', __name__)

# Module-level audit service instance (in-memory, shared across requests)
_audit_service = AuditService()
_face_verifier = FaceVerifier()
logger = logging.getLogger(__name__)


def _safe_processing_error(error):
    """Return only known user-facing processing errors from stored results."""
    safe_errors = {
        "Document OCR failed",
        "OCR completed but returned no readable text",
        "Document classification was unknown",
        "Document classification confidence is below the minimum threshold",
        "Document processing failed",
    }
    return error if error in safe_errors else (
        "Document processing failed" if error else None
    )


def _allowed_file(filename, allowed_extensions):
    """
    Check if a filename has an allowed extension.

    Args:
        filename: The original filename from the upload.
        allowed_extensions: Set of allowed lowercase extensions.

    Returns:
        True if the file extension is in the allowed set, False otherwise.
    """
    if '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in allowed_extensions


@document_bp.route('/upload', methods=['POST'])
def upload_document():
    """
    Upload a document image for future screening.

    Accepts multipart/form-data with a 'file' field.

    Returns:
        JSON response with document_id, status, and filename.
        HTTP 400 for invalid requests.
        HTTP 500 for server errors.
    """
    # --- Validate that a file was provided ---
    if 'file' not in request.files:
        return jsonify({"error": "No file provided", "field": "file"}), 400

    file = request.files['file']

    if file.filename is None or file.filename.strip() == '':
        return jsonify({"error": "No file selected"}), 400

    original_filename = file.filename
    safe_original = secure_filename(original_filename)

    extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
    if extension == 'pdf':
        return jsonify({
            "error": "PDF uploads are not supported",
            "detail": "Upload a PNG, JPG, JPEG, BMP, or TIFF image instead.",
        }), 400

    # --- Validate file extension ---
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', set())
    if not _allowed_file(original_filename, allowed_extensions):
        allowed_list = ', '.join(sorted(allowed_extensions))
        return jsonify({
            "error": "File type not allowed",
            "allowed_types": allowed_list,
        }), 400

    # --- Generate unique document ID and safe filename ---
    document_id = str(uuid.uuid4())
    # Prefix with document ID to guarantee uniqueness on disk
    stored_filename = f"{document_id}_{safe_original}"
    file_extension = extension

    # --- Save file to disk ---
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, stored_filename)

    try:
        file.save(file_path)
        file_size = os.path.getsize(file_path)
    except Exception:
        logger.exception("Document upload file save failed")
        return jsonify({"error": "Unable to save the uploaded document"}), 500

    # --- Persist metadata to database ---
    upload_timestamp = datetime.now(timezone.utc).isoformat()

    document_record = {
        "id": document_id,
        "original_filename": safe_original,
        "stored_filename": stored_filename,
        "file_type": file_extension,
        "file_size_bytes": file_size,
        "upload_timestamp": upload_timestamp,
        "processing_status": "uploaded",
    }

    try:
        db_path = current_app.config['DATABASE_PATH']
        repo = DocumentRepository(db_path)
        repo.save_document(document_record)
    except Exception:
        logger.exception("Document metadata persistence failed")
        # Clean up the saved file if database write fails
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                logger.exception("Uploaded document cleanup failed")
        return jsonify({"error": "Unable to save document metadata"}), 500

    # --- Log audit event (best-effort; upload is still valid if audit fails) ---
    audit_result = None
    audit_error = None
    try:
        audit_result = _audit_service.log_event("DOCUMENT_UPLOAD", {
            "document_id": document_id,
            "file_type": file_extension,
            "action": "upload",
        })
    except Exception:
        # Audit failure is logged but does not fail the upload
        logger.exception("Document upload audit logging failed")
        audit_error = True

    # --- Build response (no absolute filesystem paths exposed) ---
    response = {
        "document_id": document_id,
        "status": "uploaded",
        "filename": safe_original,
        "file_type": file_extension,
        "file_size_bytes": file_size,
        "upload_timestamp": upload_timestamp,
    }

    if audit_result:
        response["audit"] = {
            "block_index": audit_result["block_index"],
            "block_hash": audit_result["block_hash"],
        }
    if audit_error:
        response["audit_warning"] = "Audit logging failed; upload was still saved."

    return jsonify(response), 200


@document_bp.route('/<document_id>/status', methods=['GET'])
def document_status(document_id):
    """
    Get the current status and metadata of an uploaded document.

    Args:
        document_id: UUID of the document (from URL path).

    Returns:
        JSON response with document metadata.
        HTTP 404 if document not found.
    """
    try:
        db_path = current_app.config['DATABASE_PATH']
        repo = DocumentRepository(db_path)
        doc = repo.get_document(document_id)
    except Exception:
        logger.exception("Document status lookup failed")
        return jsonify({"error": "Unable to retrieve document status"}), 500

    if doc is None:
        return jsonify({"error": "Document not found"}), 404

    # Return metadata without exposing server filesystem paths
    return jsonify({
        "document_id": doc["id"],
        "filename": secure_filename(doc["original_filename"]),
        "file_type": doc["file_type"],
        "file_size_bytes": doc["file_size_bytes"],
        "upload_timestamp": doc["upload_timestamp"],
        "processing_status": doc["processing_status"],
    }), 200


@document_bp.route('/<document_id>/process', methods=['POST'])
def process_document(document_id):
    """
    Trigger OCR and classification processing on an uploaded document.

    This endpoint runs the document through the processing pipeline:
    OCR text extraction → field extraction → document type classification.

    Args:
        document_id: UUID of the uploaded document (from URL path).

    Returns:
        JSON response with processing results.
        HTTP 404 if document not found.
        HTTP 500 if processing fails unexpectedly.
    """
    from backend.services.document_services import DocumentProcessingService

    try:
        db_path = current_app.config['DATABASE_PATH']
        upload_folder = current_app.config['UPLOAD_FOLDER']
        service = DocumentProcessingService(
            db_path,
            upload_folder,
            current_app.config['SYNTHETIC_IDENTITY_DB_PATH'],
        )
        result = service.process_document(document_id)
    except Exception:
        logger.exception("Document processing request failed")
        return jsonify({"error": "Document processing failed"}), 500

    if not result.get("success"):
        error_msg = result.get("error", "Processing failed")
        # Only "Document not found" → 404; all other failures → 422
        if error_msg == "Document not found":
            return jsonify(result), 404
        if error_msg == "Document is already being processed":
            return jsonify(result), 409
        return jsonify(result), 422

    return jsonify(result), 200


@document_bp.route('/<document_id>/result', methods=['GET'])
def get_processing_result(document_id):
    """
    Retrieve stored processing results for a document.

    Args:
        document_id: UUID of the document (from URL path).

    Returns:
        JSON response with processing results.
        HTTP 404 if document or results not found.
    """
    import json as json_module

    try:
        db_path = current_app.config['DATABASE_PATH']
        repo = DocumentRepository(db_path)

        doc = repo.get_document(document_id)
        if doc is None:
            return jsonify({"error": "Document not found"}), 404

        result = repo.get_processing_result(document_id)
        if result is None:
            return jsonify({
                "error": "No processing results available",
                "document_id": document_id,
                "processing_status": doc["processing_status"],
            }), 404

        # Parse extracted_fields from JSON string and keep only redacted values.
        fields = {}
        if result.get("extracted_fields"):
            try:
                fields = json_module.loads(result["extracted_fields"])
            except (json_module.JSONDecodeError, TypeError):
                fields = {}

        return jsonify({
            "document_id": document_id,
            "document_type": result.get("document_type"),
            "confidence": result.get("confidence"),
            "extracted_text": None,
            "extracted_fields": fields,
            "ocr_engine": result.get("ocr_engine"),
            "ocr_confidence": result.get("ocr_confidence"),
            "processing_error": _safe_processing_error(result.get("processing_error")),
            "identity_match": result.get("identity_match"),
            "processed_at": result.get("processed_at"),
            "processing_status": doc["processing_status"],
        }), 200

    except Exception:
        logger.exception("Document result lookup failed")
        return jsonify({"error": "Unable to retrieve document results"}), 500


@document_bp.route('/<document_id>/compare-face', methods=['POST'])
def compare_face(document_id):
    """Compare a client-captured face with the document's detected portrait."""
    if 'face' not in request.files:
        return jsonify({"error": "No live face image provided", "field": "face"}), 400

    face_file = request.files['face']
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', set())
    if not _allowed_file(face_file.filename or '', allowed_extensions):
        return jsonify({
            "error": "Face image type not allowed",
            "allowed_types": ', '.join(sorted(allowed_extensions)),
        }), 400

    live_image = face_file.read()
    if not live_image:
        return jsonify({"error": "Live face image is empty"}), 400

    try:
        db_path = current_app.config['DATABASE_PATH']
        repo = DocumentRepository(db_path)
        doc = repo.get_document(document_id)
        if doc is None:
            return jsonify({"error": "Document not found"}), 404

        file_path = os.path.join(
            current_app.config['UPLOAD_FOLDER'],
            doc['stored_filename'],
        )
        if not os.path.isfile(file_path):
            return jsonify({"error": "Uploaded file not found on disk"}), 404

        result = _face_verifier.compare(
            file_path,
            live_image,
            threshold=current_app.config['FACE_MATCH_THRESHOLD'],
        )
        if result.get('reason', '').startswith('Face comparison unavailable:'):
            result['reason'] = 'Face comparison is unavailable.'
        result['document_id'] = document_id
        return jsonify(result), 200
    except Exception:
        logger.exception("Face comparison request failed")
        return jsonify({
            "error": "Face comparison failed",
        }), 422


