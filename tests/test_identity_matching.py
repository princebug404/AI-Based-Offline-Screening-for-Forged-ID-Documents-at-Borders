import io
import json
import os
import sqlite3
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from backend.app import create_app
from backend.config import Config
from backend.services.identity_matching import SyntheticIdentityMatcher
from database.db import init_db
from database.repository import DocumentRepository


class TestSyntheticIdentityMatcher(unittest.TestCase):
    def setUp(self):
        self.matcher = SyntheticIdentityMatcher(Config.SYNTHETIC_IDENTITY_DB_PATH)

    def test_matching_record_normalizes_formatting(self):
        result = self.matcher.match(
            'passport',
            {
                'document_number': 'SYNTH P 000001',
                'name': ' sharma ',
                'date_of_birth': '1987-04-08',
            },
        )

        self.assertEqual(result['status'], 'Match')
        self.assertEqual(result['record_id'], 'SYN-PASSPORT-001')
        self.assertEqual(
            result['fields'],
            {
                'document_type': 'Match',
                'document_number': 'Match',
                'name': 'Match',
                'date_of_birth': 'Match',
            },
        )

    def test_mismatched_field_is_reported(self):
        result = self.matcher.match(
            'aadhaar',
            {
                'document_number': '1111-2222-3333',
                'name': 'DIFFERENT SYNTHETIC NAME',
                'date_of_birth': '1995',
            },
        )

        self.assertEqual(result['status'], 'Mismatch')
        self.assertEqual(result['fields']['name'], 'Mismatch')
        self.assertEqual(result['fields']['document_number'], 'Match')

    def test_known_number_with_wrong_document_type_reports_mismatch(self):
        result = self.matcher.match(
            'aadhaar',
            {
                'document_number': 'SYNTH-P-000001',
                'name': 'SHARMA',
                'date_of_birth': '08/04/1987',
            },
        )

        self.assertEqual(result['status'], 'Mismatch')
        self.assertEqual(result['fields']['document_type'], 'Mismatch')
        self.assertEqual(result['fields']['document_number'], 'Match')

    def test_missing_field_is_unavailable_and_requires_review(self):
        result = self.matcher.match(
            'aadhaar',
            {
                'document_number': '1111 2222 3333',
                'name': None,
                'date_of_birth': '1995',
            },
        )

        self.assertEqual(result['status'], 'Manual Review')
        self.assertEqual(result['fields']['name'], 'Unavailable')

    def test_unknown_record_does_not_guess(self):
        result = self.matcher.match(
            'passport',
            {
                'document_number': 'SYNTH-P-999999',
                'name': 'SHARMA',
                'date_of_birth': '08/04/1987',
            },
        )

        self.assertEqual(result['status'], 'Unknown Record')
        self.assertIsNone(result['record_id'])
        self.assertTrue(all(value == 'Unavailable' for value in result['fields'].values()))


class TestSyntheticIdentityProcessing(unittest.TestCase):
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

    def test_processing_returns_field_level_synthetic_match_without_values(self):
        image = Image.new('RGB', (240, 160), 'white')
        image_buffer = io.BytesIO()
        image.save(image_buffer, format='PNG')
        image_buffer.seek(0)
        upload = self.client.post(
            '/api/documents/upload',
            data={'file': (image_buffer, 'synthetic-passport.png')},
            content_type='multipart/form-data',
        )
        document_id = json.loads(upload.data)['document_id']

        ocr_result = {
            'success': True,
            'text': (
                'REPUBLIC OF INDIA\nPASSPORT\n'
                'Surname: SHARMA\n'
                'Date of Birth: 08/04/1987\n'
                'Passport No: SYNTH-P-000001'
            ),
            'error': None,
            'engine': 'tesseract',
            'confidence': 90.0,
        }
        with patch('backend.services.document_services.OCREngine') as mock_engine:
            mock_engine.return_value.extract_text.return_value = ocr_result
            response = self.client.post(f'/api/documents/{document_id}/process')

        payload = json.loads(response.data)
        identity_match = payload['identity_match']
        self.assertEqual(response.status_code, 200)
        self.assertEqual(identity_match['status'], 'Match')
        self.assertEqual(identity_match['record_id'], 'SYN-PASSPORT-001')
        self.assertNotIn('name', identity_match)
        self.assertNotIn('document_number', identity_match)
        self.assertNotIn('date_of_birth', identity_match)

        retrieved = self.client.get(f'/api/documents/{document_id}/result')
        self.assertEqual(retrieved.status_code, 200)
        self.assertEqual(
            json.loads(retrieved.data)['identity_match'],
            identity_match,
        )

    def test_legacy_result_without_identity_match_returns_null(self):
        image = Image.new('RGB', (240, 160), 'white')
        image_buffer = io.BytesIO()
        image.save(image_buffer, format='PNG')
        image_buffer.seek(0)
        upload = self.client.post(
            '/api/documents/upload',
            data={'file': (image_buffer, 'legacy-result.png')},
            content_type='multipart/form-data',
        )
        document_id = json.loads(upload.data)['document_id']
        repository = DocumentRepository(self.app.config['DATABASE_PATH'])
        repository.save_processing_result_and_status({
            'document_id': document_id,
            'document_type': 'unknown',
            'confidence': 0.0,
            'extracted_fields': '{}',
            'processing_error': 'legacy result',
        }, 'failed')

        retrieved = self.client.get(f'/api/documents/{document_id}/result')
        self.assertEqual(retrieved.status_code, 200)
        self.assertIsNone(json.loads(retrieved.data)['identity_match'])

    def test_existing_schema_is_migrated_with_nullable_identity_match(self):
        legacy_db = os.path.join(self.test_dir, 'legacy.db')
        connection = sqlite3.connect(legacy_db)
        connection.executescript("""
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size_bytes INTEGER,
                upload_timestamp TEXT NOT NULL,
                processing_status TEXT NOT NULL DEFAULT 'uploaded'
            );
            CREATE TABLE processing_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL UNIQUE,
                document_type TEXT,
                confidence REAL,
                extracted_text TEXT,
                extracted_fields TEXT,
                ocr_engine TEXT,
                ocr_confidence REAL,
                processing_error TEXT,
                processed_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );
        """)
        connection.commit()
        connection.close()

        init_db(legacy_db)
        connection = sqlite3.connect(legacy_db)
        columns = {
            row[1]
            for row in connection.execute('PRAGMA table_info(processing_results)')
        }
        connection.close()
        self.assertIn('identity_match', columns)
