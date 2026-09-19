# SIH26188 - API route tests

import io
import os
import json
import sqlite3
import shutil
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app import create_app
from database.db import get_connection, init_db
from database.repository import DocumentRepository


class TestHealthEndpoint(unittest.TestCase):
    """Tests for GET /api/health."""

    def setUp(self):
        """Create a test app with a temporary database and upload folder."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.upload_folder = os.path.join(self.test_dir, 'uploads')

        self.app = create_app({
            'TESTING': True,
            'DATABASE_PATH': self.db_path,
            'UPLOAD_FOLDER': self.upload_folder,
            'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'bmp', 'tiff'},
            'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
        })
        self.client = self.app.test_client()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_health_returns_200(self):
        """Health endpoint should return HTTP 200."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)

    def test_health_returns_json(self):
        """Health endpoint should return valid JSON with expected fields."""
        response = self.client.get('/api/health')
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'running')
        self.assertEqual(data['service'], 'SIH26188')
        self.assertIn('timestamp', data)


class TestDebugConfiguration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'TESTING': True,
            'DATABASE_PATH': os.path.join(self.test_dir, 'test.db'),
            'UPLOAD_FOLDER': os.path.join(self.test_dir, 'uploads'),
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch.dict(os.environ, {'FLASK_DEBUG': ''}, clear=False)
    def test_debug_is_disabled_by_default(self):
        app = create_app(self.config)
        self.assertFalse(app.debug)

    @patch.dict(os.environ, {'FLASK_DEBUG': 'true'}, clear=False)
    def test_debug_requires_explicit_opt_in(self):
        app = create_app(self.config)
        self.assertTrue(app.debug)

    @patch.dict(os.environ, {'FLASK_DEBUG': 'true'}, clear=False)
    def test_config_override_can_disable_debug(self):
        config = dict(self.config)
        config['DEBUG'] = False
        app = create_app(config)
        self.assertFalse(app.debug)


class TestDocumentUpload(unittest.TestCase):
    """Tests for POST /api/documents/upload."""

    def setUp(self):
        """Create a test app with a temporary database and upload folder."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.upload_folder = os.path.join(self.test_dir, 'uploads')

        self.app = create_app({
            'TESTING': True,
            'DATABASE_PATH': self.db_path,
            'UPLOAD_FOLDER': self.upload_folder,
            'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'bmp', 'tiff'},
            'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
        })
        self.client = self.app.test_client()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_upload_success(self):
        """Uploading a valid PNG file should succeed."""
        data = {
            'file': (io.BytesIO(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100), 'test_doc.png'),
        }
        response = self.client.post(
            '/api/documents/upload',
            data=data,
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('document_id', result)
        self.assertEqual(result['status'], 'uploaded')
        self.assertEqual(result['filename'], 'test_doc.png')
        self.assertEqual(result['file_type'], 'png')
        self.assertIn('file_size_bytes', result)
        self.assertIn('upload_timestamp', result)

    def test_upload_returns_document_id(self):
        """Upload response should contain a valid UUID document ID."""
        data = {
            'file': (io.BytesIO(b'fake image data'), 'photo.jpg'),
        }
        response = self.client.post(
            '/api/documents/upload',
            data=data,
            content_type='multipart/form-data',
        )
        result = json.loads(response.data)
        doc_id = result['document_id']
        # UUID4 format: 8-4-4-4-12 hex characters
        self.assertEqual(len(doc_id), 36)
        self.assertEqual(doc_id.count('-'), 4)

    def test_upload_missing_file(self):
        """POST without a file field should return 400."""
        response = self.client.post(
            '/api/documents/upload',
            data={},
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 400)
        result = json.loads(response.data)
        self.assertIn('error', result)

    def test_upload_empty_filename(self):
        """POST with an empty filename should return 400."""
        data = {
            'file': (io.BytesIO(b'data'), ''),
        }
        response = self.client.post(
            '/api/documents/upload',
            data=data,
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 400)

    def test_upload_invalid_type(self):
        """Uploading a file with a disallowed extension should return 400."""
        data = {
            'file': (io.BytesIO(b'MZ' + b'\x00' * 100), 'malware.exe'),
        }
        response = self.client.post(
            '/api/documents/upload',
            data=data,
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 400)
        result = json.loads(response.data)
        self.assertIn('error', result)
        self.assertIn('allowed_types', result)

    def test_upload_valid_pdf_is_rejected(self):
        """A valid PDF must be rejected because PDF OCR is not supported."""
        valid_pdf = (
            b'%PDF-1.4\n'
            b'1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n'
            b'2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\n'
            b'trailer<</Root 1 0 R>>\n%%EOF\n'
        )
        response = self.client.post(
            '/api/documents/upload',
            data={'file': (io.BytesIO(valid_pdf), 'document.pdf')},
            content_type='multipart/form-data',
        )

        self.assertEqual(response.status_code, 400)
        result = json.loads(response.data)
        self.assertEqual(result['error'], 'PDF uploads are not supported')

    def test_upload_corrupted_pdf_is_rejected(self):
        """A corrupted PDF must be rejected before it reaches OCR processing."""
        response = self.client.post(
            '/api/documents/upload',
            data={'file': (io.BytesIO(b'not a real pdf'), 'corrupted.pdf')},
            content_type='multipart/form-data',
        )

        self.assertEqual(response.status_code, 400)
        result = json.loads(response.data)
        self.assertEqual(result['error'], 'PDF uploads are not supported')

    def test_unique_document_ids(self):
        """Two separate uploads should produce different document IDs."""
        data1 = {
            'file': (io.BytesIO(b'image data 1'), 'doc1.png'),
        }
        data2 = {
            'file': (io.BytesIO(b'image data 2'), 'doc2.png'),
        }
        resp1 = self.client.post('/api/documents/upload', data=data1, content_type='multipart/form-data')
        resp2 = self.client.post('/api/documents/upload', data=data2, content_type='multipart/form-data')

        id1 = json.loads(resp1.data)['document_id']
        id2 = json.loads(resp2.data)['document_id']
        self.assertNotEqual(id1, id2)

    def test_upload_file_saved_to_disk(self):
        """Uploaded file should be physically saved in the upload folder."""
        file_content = b'test file content for disk check'
        data = {
            'file': (io.BytesIO(file_content), 'disk_test.jpg'),
        }
        response = self.client.post(
            '/api/documents/upload',
            data=data,
            content_type='multipart/form-data',
        )
        result = json.loads(response.data)
        doc_id = result['document_id']

        # Find the saved file in the upload folder
        saved_files = os.listdir(self.upload_folder)
        matching = [f for f in saved_files if f.startswith(doc_id)]
        self.assertEqual(len(matching), 1, "Exactly one file should be saved for this upload")

    def test_upload_no_absolute_paths_in_response(self):
        """API response must not expose absolute filesystem paths."""
        data = {
            'file': (io.BytesIO(b'some data'), 'secret_doc.png'),
        }
        response = self.client.post(
            '/api/documents/upload',
            data=data,
            content_type='multipart/form-data',
        )
        response_text = response.data.decode('utf-8')
        # Should not contain drive letters or Unix root paths
        self.assertNotIn(':\\', response_text)
        self.assertNotIn(self.upload_folder, response_text)

    @patch('backend.routes.document_routes.DocumentRepository.save_document',
           side_effect=RuntimeError('database path C:\\private\\records.db'))
    def test_upload_database_error_does_not_expose_exception(self, _save_document):
        response = self.client.post(
            '/api/documents/upload',
            data={'file': (io.BytesIO(b'image data'), 'private.png')},
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 500)
        response_text = response.data.decode()
        self.assertIn('Unable to save document metadata', response_text)
        self.assertNotIn('private', response_text)
        self.assertNotIn('database path', response_text)


class TestDocumentStatus(unittest.TestCase):
    """Tests for GET /api/documents/<document_id>/status."""

    def setUp(self):
        """Create a test app with a temporary database and upload folder."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.upload_folder = os.path.join(self.test_dir, 'uploads')

        self.app = create_app({
            'TESTING': True,
            'DATABASE_PATH': self.db_path,
            'UPLOAD_FOLDER': self.upload_folder,
            'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'bmp', 'tiff'},
            'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
        })
        self.client = self.app.test_client()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_document_status_after_upload(self):
        """Status endpoint should return metadata for an uploaded document."""
        # Upload a document first
        data = {
            'file': (io.BytesIO(b'test content'), 'status_test.png'),
        }
        upload_resp = self.client.post(
            '/api/documents/upload',
            data=data,
            content_type='multipart/form-data',
        )
        doc_id = json.loads(upload_resp.data)['document_id']

        # Check status
        status_resp = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(status_resp.status_code, 200)
        result = json.loads(status_resp.data)
        self.assertEqual(result['document_id'], doc_id)
        self.assertEqual(result['processing_status'], 'uploaded')
        self.assertEqual(result['filename'], 'status_test.png')

    def test_document_status_not_found(self):
        """Status endpoint should return 404 for unknown document ID."""
        response = self.client.get('/api/documents/nonexistent-id-12345/status')
        self.assertEqual(response.status_code, 404)
        result = json.loads(response.data)
        self.assertIn('error', result)

    @patch('backend.routes.document_routes.DocumentRepository.get_document',
           side_effect=RuntimeError('sqlite file C:\\private\\status.db'))
    def test_status_database_error_does_not_expose_exception(self, _get_document):
        response = self.client.get('/api/documents/sample/status')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json['error'], 'Unable to retrieve document status')
        self.assertNotIn('status.db', response.data.decode())

    @patch('backend.services.document_services.OCREngine')
    def test_failed_processing_status_matches_result(self, MockOCREngine):
        """Failed processing should be reflected consistently by status and result APIs."""
        MockOCREngine.return_value.extract_text.return_value = {
            "success": True, "text": "", "error": None,
            "engine": "tesseract", "confidence": None,
        }
        data = {
            'file': (io.BytesIO(b'image data'), 'unreadable.png'),
        }
        upload = self.client.post(
            '/api/documents/upload', data=data, content_type='multipart/form-data'
        )
        doc_id = json.loads(upload.data)['document_id']

        process = self.client.post(f'/api/documents/{doc_id}/process')
        self.assertEqual(process.status_code, 422)
        self.assertEqual(json.loads(process.data)['processing_status'], 'failed')

        status = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(json.loads(status.data)['processing_status'], 'failed')
        result = self.client.get(f'/api/documents/{doc_id}/result')
        result_data = json.loads(result.data)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result_data['processing_status'], 'failed')
        self.assertIn('no readable text', result_data['processing_error'])


class TestDatabasePersistence(unittest.TestCase):
    """Tests verifying document metadata survives app restarts."""

    def setUp(self):
        """Create a persistent test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'persist_test.db')
        self.upload_folder = os.path.join(self.test_dir, 'uploads')
        self.config = {
            'TESTING': True,
            'DATABASE_PATH': self.db_path,
            'UPLOAD_FOLDER': self.upload_folder,
            'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'bmp', 'tiff'},
            'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
        }

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_persistence_across_app_restart(self):
        """Document metadata should persist when a new app instance is created."""
        # Upload with first app instance
        app1 = create_app(self.config)
        client1 = app1.test_client()
        data = {
            'file': (io.BytesIO(b'persistent data'), 'persist.jpg'),
        }
        resp = client1.post('/api/documents/upload', data=data, content_type='multipart/form-data')
        doc_id = json.loads(resp.data)['document_id']

        # Create a new app instance (simulates restart) with the same DB
        app2 = create_app(self.config)
        client2 = app2.test_client()
        status_resp = client2.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(status_resp.status_code, 200)
        result = json.loads(status_resp.data)
        self.assertEqual(result['document_id'], doc_id)
        self.assertEqual(result['filename'], 'persist.jpg')


class TestRouteErrorHardening(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.app = create_app({
            'TESTING': True,
            'DATABASE_PATH': os.path.join(self.test_dir, 'test.db'),
            'UPLOAD_FOLDER': os.path.join(self.test_dir, 'uploads'),
            'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'bmp', 'tiff'},
        })
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('backend.services.document_services.DocumentProcessingService',
           side_effect=RuntimeError('model path C:\\private\\model.onnx'))
    def test_process_route_does_not_expose_exception(self, _service):
        response = self.client.post('/api/documents/sample/process')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json['error'], 'Document processing failed')
        self.assertNotIn('model.onnx', response.data.decode())

    @patch('backend.routes.document_routes.DocumentRepository.get_document',
           side_effect=RuntimeError('sqlite error at C:\\private\\results.db'))
    def test_result_route_does_not_expose_exception(self, _get_document):
        response = self.client.get('/api/documents/sample/result')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json['error'], 'Unable to retrieve document results')
        self.assertNotIn('results.db', response.data.decode())

    def test_face_route_does_not_expose_comparison_exception(self):
        upload = self.client.post(
            '/api/documents/upload',
            data={'file': (io.BytesIO(b'image data'), 'document.png')},
            content_type='multipart/form-data',
        )
        document_id = upload.json['document_id']
        with patch(
            'backend.routes.document_routes._face_verifier.compare',
            side_effect=RuntimeError('embedding path C:\\private\\model.onnx'),
        ):
            response = self.client.post(
                f'/api/documents/{document_id}/compare-face',
                data={'face': (io.BytesIO(b'face bytes'), 'face.png')},
                content_type='multipart/form-data',
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json['error'], 'Face comparison failed')
        self.assertNotIn('model.onnx', response.data.decode())


class TestDatabaseIntegrity(unittest.TestCase):
    """Tests for SQLite foreign-key enforcement and processing-result integrity."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'integrity_test.db')
        init_db(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_new_connection_enables_foreign_keys(self):
        """Every new database connection should enforce foreign keys."""
        conn = get_connection(self.db_path)
        try:
            enabled = conn.execute('PRAGMA foreign_keys').fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(enabled, 1)

    def test_processing_result_requires_existing_document(self):
        """An orphan processing result should be rejected by SQLite."""
        repository = DocumentRepository(self.db_path)
        with self.assertRaises(sqlite3.IntegrityError):
            repository.save_processing_result({
                'document_id': 'missing-document',
                'document_type': 'unknown',
                'confidence': 0.0,
                'extracted_fields': '{}',
                'processing_error': 'test failure',
            })


class TestAuditFailure(unittest.TestCase):
    """Tests for upload behavior when blockchain audit logging fails."""

    def setUp(self):
        """Create a test app with a temporary database and upload folder."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.upload_folder = os.path.join(self.test_dir, 'uploads')

        self.app = create_app({
            'TESTING': True,
            'DATABASE_PATH': self.db_path,
            'UPLOAD_FOLDER': self.upload_folder,
            'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'bmp', 'tiff'},
            'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
        })
        self.client = self.app.test_client()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('backend.routes.document_routes._audit_service.log_event',
           side_effect=RuntimeError("Ledger write failed"))
    def test_upload_succeeds_with_audit_warning(self, mock_log_event):
        """Upload should return 200 with audit_warning when audit logging fails."""
        data = {
            'file': (io.BytesIO(b'image content'), 'audit_fail_test.png'),
        }
        response = self.client.post(
            '/api/documents/upload',
            data=data,
            content_type='multipart/form-data',
        )

        # Upload itself should still succeed
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)

        # Core upload fields must be present
        self.assertIn('document_id', result)
        self.assertEqual(result['status'], 'uploaded')
        self.assertEqual(result['filename'], 'audit_fail_test.png')

        # The audit block should NOT be present
        self.assertNotIn('audit', result)

        # The audit warning MUST be present
        self.assertIn('audit_warning', result)
        self.assertEqual(
            result['audit_warning'],
            "Audit logging failed; upload was still saved.",
        )

        # Verify the mock was actually called
        mock_log_event.assert_called_once()

    @patch('backend.routes.document_routes._audit_service.log_event',
           side_effect=RuntimeError("Ledger write failed"))
    def test_database_persists_despite_audit_failure(self, mock_log_event):
        """Document metadata must be in the database even if audit logging failed."""
        data = {
            'file': (io.BytesIO(b'persist despite audit fail'), 'db_check.jpg'),
        }
        response = self.client.post(
            '/api/documents/upload',
            data=data,
            content_type='multipart/form-data',
        )
        result = json.loads(response.data)
        doc_id = result['document_id']

        # Query the status endpoint to confirm DB persistence
        status_resp = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(status_resp.status_code, 200)
        status_result = json.loads(status_resp.data)
        self.assertEqual(status_result['document_id'], doc_id)
        self.assertEqual(status_result['filename'], 'db_check.jpg')
        self.assertEqual(status_result['processing_status'], 'uploaded')

    @patch('backend.routes.document_routes._audit_service.log_event',
           side_effect=RuntimeError("Ledger write failed"))
    def test_file_saved_despite_audit_failure(self, mock_log_event):
        """Uploaded file must exist on disk even if audit logging failed."""
        data = {
            'file': (io.BytesIO(b'file on disk check'), 'disk_audit_fail.png'),
        }
        response = self.client.post(
            '/api/documents/upload',
            data=data,
            content_type='multipart/form-data',
        )
        result = json.loads(response.data)
        doc_id = result['document_id']

        # Verify file exists in the upload folder
        saved_files = os.listdir(self.upload_folder)
        matching = [f for f in saved_files if f.startswith(doc_id)]
        self.assertEqual(len(matching), 1)


if __name__ == '__main__':
    unittest.main()

