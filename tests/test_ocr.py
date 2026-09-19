# SIH26188 - OCR, field extraction, classification, and processing API tests

import io
import os
import json
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image, ImageDraw, ImageFont

from backend.app import create_app
from src.ocr.field_extraction import FieldExtractor
from src.vision.document_classfication import DocumentClassifier


def _create_test_image(text_lines=None, size=(400, 300)):
    """
    Create a synthetic document image with text drawn on it.

    Args:
        text_lines: List of strings to draw. Defaults to a sample passport-like layout.
        size: (width, height) tuple.

    Returns:
        PIL Image object.
    """
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)

    if text_lines is None:
        text_lines = [
            "REPUBLIC OF INDIA",
            "PASSPORT",
            "Surname: SHARMA",
            "Given Name: RAJESH KUMAR",
            "Date of Birth: 15/03/1990",
            "Date of Expiry: 14/03/2030",
            "Passport No: J8369854",
            "Nationality: INDIAN",
        ]

    y = 10
    for line in text_lines:
        draw.text((10, y), line, fill='black')
        y += 25

    return img


def _save_test_image(image, directory, filename='test_doc.png'):
    """Save a PIL image to a directory and return the path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    image.save(path)
    return path


# ============================================================
# Field Extraction Tests (no OCR dependency)
# ============================================================

class TestFieldExtractor(unittest.TestCase):
    """Tests for the regex-based field extractor."""

    def setUp(self):
        self.extractor = FieldExtractor()

    def test_extract_name_from_labeled_text(self):
        """Should extract name from 'Name: ...' pattern."""
        text = "Surname: SHARMA\nGiven Name: RAJESH KUMAR\nDate of Birth: 01/01/1990"
        fields = self.extractor.extract_fields(text)
        # Should find at least one name field
        self.assertIsNotNone(fields['name'])
        self.assertIn('extraction_method', fields)
        self.assertEqual(fields['extraction_method'], 'regex_baseline')

    def test_extract_dob(self):
        """Should extract date of birth from labeled text."""
        text = "Date of Birth: 15/03/1990\nPlace of Birth: Delhi"
        fields = self.extractor.extract_fields(text)
        self.assertIsNotNone(fields['date_of_birth'])
        self.assertIn('15', fields['date_of_birth'])
        self.assertIn('1990', fields['date_of_birth'])

    def test_extract_dob_alternate_label(self):
        """Should extract DOB with 'DOB' label."""
        text = "DOB: 25-12-1985\nAddress: Mumbai"
        fields = self.extractor.extract_fields(text)
        self.assertIsNotNone(fields['date_of_birth'])

    def test_extract_document_number(self):
        """Should extract alphanumeric document numbers."""
        text = "Passport No: J8369854\nNationality: INDIAN"
        fields = self.extractor.extract_fields(text)
        self.assertIsNotNone(fields['document_number'])
        self.assertEqual(fields['document_number'], 'J8369854')

    def test_extract_synthetic_aadhaar_fields(self):
        """Should extract fictional Aadhaar fields that are visibly labeled."""
        text = (
            "Government of India\nAadhaar Card\n"
            "Name: KAVYA MEHTA\nYear of Birth: 1995\n"
            "Aadhaar Number: 9999 8888 7777"
        )
        fields = self.extractor.extract_fields(text, document_type='aadhaar')

        self.assertEqual(fields['name'], 'KAVYA MEHTA')
        self.assertEqual(fields['date_of_birth'], '1995')
        self.assertEqual(fields['document_number'], '9999 8888 7777')
        self.assertIsNone(fields['expiry_date'])

    def test_extract_synthetic_passport_number_with_hyphens(self):
        """Should preserve a fictional labeled passport number with separators."""
        text = "REPUBLIC OF INDIA\nPASSPORT\nPassport No: DEMO-P-000001"
        fields = self.extractor.extract_fields(text, document_type='passport')
        self.assertEqual(fields['document_number'], 'DEMO-P-000001')

    def test_extract_passport_number_after_ocr_label_tokens(self):
        """Should skip OCR label tokens before the fictional passport number."""
        text = "Passport No.\nPASSPORT P IND DEMO-P-000001"
        fields = self.extractor.extract_fields(text, document_type='passport')
        self.assertEqual(fields['document_number'], 'DEMO-P-000001')

    def test_unreadable_synthetic_fields_remain_none(self):
        """Unreadable Aadhaar values should not be invented from nearby text."""
        text = "AADHAAR CARD\nName: KAVYA MEHTA\nYear of Birth: unreadable\nAadhaar Number: unavailable"
        fields = self.extractor.extract_fields(text, document_type='aadhaar')
        self.assertEqual(fields['name'], 'KAVYA MEHTA')
        self.assertIsNone(fields['date_of_birth'])
        self.assertIsNone(fields['document_number'])

    def test_extract_expiry_date(self):
        """Should extract expiry date from labeled text."""
        text = "Date of Expiry: 14/03/2030\nAuthority: MEA"
        fields = self.extractor.extract_fields(text)
        self.assertIsNotNone(fields['expiry_date'])
        self.assertIn('2030', fields['expiry_date'])

    def test_empty_text_returns_null_fields(self):
        """Empty text should return all None fields."""
        fields = self.extractor.extract_fields("")
        self.assertIsNone(fields['name'])
        self.assertIsNone(fields['date_of_birth'])
        self.assertIsNone(fields['document_number'])
        self.assertIsNone(fields['expiry_date'])
        self.assertEqual(fields['field_count'], 0)

    def test_none_text_returns_null_fields(self):
        """None text should return all None fields."""
        fields = self.extractor.extract_fields(None)
        self.assertEqual(fields['field_count'], 0)

    def test_no_invented_fields(self):
        """Unrecognizable text should NOT produce invented field values."""
        text = "Random noise abc xyz 12345 garbage text !@#$%"
        fields = self.extractor.extract_fields(text)
        # Name should be None since there's no "Name:" label
        self.assertIsNone(fields['name'])
        # DOB should be None since there's no DOB label
        self.assertIsNone(fields['date_of_birth'])

    def test_field_count_accuracy(self):
        """field_count should match the number of non-None primary fields."""
        text = "Name: John Doe\nDate of Birth: 01/01/2000\nExpiry Date: 31/12/2030"
        fields = self.extractor.extract_fields(text)
        non_none = sum(1 for k in ['name', 'date_of_birth', 'document_number', 'expiry_date']
                       if fields[k] is not None)
        self.assertEqual(fields['field_count'], non_none)

    def test_all_dates_extraction(self):
        """all_dates should find all date-like patterns."""
        text = "DOB: 15/03/1990\nIssue: 20/04/2020\nExpiry: 19/04/2030"
        fields = self.extractor.extract_fields(text)
        self.assertGreaterEqual(len(fields['all_dates']), 2)


# ============================================================
# Document Classification Tests (no OCR dependency)
# ============================================================

class TestDocumentClassifier(unittest.TestCase):
    """Tests for the keyword-based document classifier."""

    def setUp(self):
        self.classifier = DocumentClassifier()

    def test_classify_passport(self):
        """Text with passport keywords should classify as passport."""
        text = "REPUBLIC OF INDIA\nPASSPORT\nSurname: DOE\nGiven Name: JOHN\nDate of Birth: 01/01/1990"
        result = self.classifier.classify(text)
        self.assertEqual(result['document_type'], 'passport')
        self.assertGreater(result['confidence'], 0.0)
        self.assertEqual(result['method'], 'keyword_baseline')

    def test_classify_aadhaar(self):
        """Text with Aadhaar keywords should classify as aadhaar."""
        text = "Government of India\nAadhaar\nUnique Identification Authority\nDOB: 01/01/1990\nMale"
        result = self.classifier.classify(text)
        self.assertEqual(result['document_type'], 'aadhaar')

    def test_classify_aadhaar_from_distinctive_authority_label(self):
        """The authority label should identify Aadhaar without relying on one token."""
        text = "Unique Identification Authority of India\nYear of Birth: 1995"
        result = self.classifier.classify(text)
        self.assertEqual(result['document_type'], 'aadhaar')

    def test_classify_pan_card(self):
        """Text with PAN keywords should classify as pan_card."""
        text = "INCOME TAX DEPARTMENT\nPermanent Account Number\nGovt of India\nABCDE1234F"
        result = self.classifier.classify(text)
        self.assertEqual(result['document_type'], 'pan_card')

    def test_classify_driving_license(self):
        """Text with driving license keywords should classify as driving_license."""
        text = "DRIVING LICENCE\nTransport Department\nValid Till: 2030\nClass of Vehicle: LMV"
        result = self.classifier.classify(text)
        self.assertEqual(result['document_type'], 'driving_license')

    def test_classify_unknown(self):
        """Unrecognizable text should classify as unknown."""
        text = "Random unrelated text about weather forecast and sports news"
        result = self.classifier.classify(text)
        self.assertEqual(result['document_type'], 'unknown')
        self.assertEqual(result['confidence'], 0.0)

    def test_weak_substring_matches_are_ignored(self):
        """Keywords inside larger words should not classify the document."""
        result = self.classifier.classify(
            "passporting voterish aadhaarish permanentaccountnumber"
        )
        self.assertEqual(result['document_type'], 'unknown')
        self.assertEqual(result['confidence'], 0.0)

    def test_distinctive_keyword_matches_are_supported(self):
        """Exact distinctive document terms should classify successfully."""
        for text, expected_type in (
            ("PASSPORT", "passport"),
            ("AADHAAR", "aadhaar"),
            ("DRIVING LICENSE", "driving_license"),
            ("ELECTION COMMISSION", "voter_id"),
        ):
            with self.subTest(text=text):
                result = self.classifier.classify(text)
                self.assertEqual(result['document_type'], expected_type)
                self.assertGreaterEqual(result['confidence'], 0.15)

    def test_ambiguous_text_returns_unknown(self):
        """Equally supported document types should not be guessed."""
        classifier = DocumentClassifier({
            "type_a": {"keywords": ["alpha"], "strong": ["alpha"]},
            "type_b": {"keywords": ["gamma"], "strong": ["gamma"]},
        })
        result = classifier.classify("alpha gamma")
        self.assertEqual(result['document_type'], 'unknown')
        self.assertEqual(result['confidence'], 0.0)

    def test_classify_empty_text(self):
        """Empty text should classify as unknown."""
        result = self.classifier.classify("")
        self.assertEqual(result['document_type'], 'unknown')
        self.assertEqual(result['confidence'], 0.0)

    def test_confidence_threshold_boundaries(self):
        """Scores below the threshold fail, while an exact threshold passes."""
        below_threshold = DocumentClassifier({
            "sample": {"keywords": [f"term{i}" for i in range(10)]}
        }).classify("term0")
        self.assertEqual(below_threshold['document_type'], 'unknown')
        self.assertEqual(below_threshold['confidence'], 0.0)

        at_threshold = DocumentClassifier({
            "sample": {"keywords": [f"term{i}" for i in range(20)]}
        }).classify("term0 term1 term2")
        self.assertEqual(at_threshold['document_type'], 'sample')
        self.assertEqual(at_threshold['confidence'], 0.15)

    def test_classify_none_text(self):
        """None text should classify as unknown."""
        result = self.classifier.classify(None)
        self.assertEqual(result['document_type'], 'unknown')

    def test_all_scores_present(self):
        """Result should contain scores for all document types."""
        text = "PASSPORT"
        result = self.classifier.classify(text)
        self.assertIn('all_scores', result)
        self.assertIsInstance(result['all_scores'], dict)

    def test_confidence_bounded(self):
        """Confidence should be between 0.0 and 1.0."""
        text = "PASSPORT REPUBLIC Surname Given Name Date of Birth Nationality"
        result = self.classifier.classify(text)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)


# ============================================================
# OCR Engine Tests
# ============================================================

class TestOCREngine(unittest.TestCase):
    """Tests for the OCR engine (handles missing Tesseract gracefully)."""

    def setUp(self):
        from src.ocr.ocr_engine import OCREngine
        self.engine = OCREngine()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_is_available_returns_dict(self):
        """is_available should return a dict with 'available' and 'detail'."""
        result = self.engine.is_available()
        self.assertIn('available', result)
        self.assertIn('detail', result)
        self.assertIsInstance(result['available'], bool)

    def test_extract_text_missing_file(self):
        """Extracting from a nonexistent file should return an error."""
        result = self.engine.extract_text('/nonexistent/path/fake.png')
        self.assertFalse(result['success'])
        self.assertIn('not found', result['error'].lower())

    def test_extract_text_returns_structured_result(self):
        """extract_text should always return a dict with expected keys."""
        # Create a minimal image
        img = Image.new('RGB', (100, 100), 'white')
        path = _save_test_image(img, self.test_dir, 'minimal.png')

        result = self.engine.extract_text(path)
        # Regardless of Tesseract availability, the result structure must be correct
        self.assertIn('success', result)
        self.assertIn('text', result)
        self.assertIn('error', result)
        self.assertIn('engine', result)
        self.assertEqual(result['engine'], 'tesseract')

    @patch('src.ocr.ocr_engine.pytesseract.image_to_string')
    @patch.object(__import__('src.ocr.ocr_engine', fromlist=['OCREngine']).OCREngine, '_get_confidence', return_value=77.6)
    def test_extract_text_retries_grayscale_when_threshold_is_empty(self, _confidence, mock_ocr):
        """A thresholding miss should retry readable grayscale OCR output."""
        mock_ocr.side_effect = ['', 'RECOVERED PASSPORT TEXT']
        img = Image.new('RGB', (100, 100), 'white')
        path = _save_test_image(img, self.test_dir, 'fallback.png')

        result = self.engine.extract_text(path)

        self.assertTrue(result['success'])
        self.assertEqual(result['text'], 'RECOVERED PASSPORT TEXT')
        self.assertEqual(result['confidence'], 77.6)
        self.assertEqual(mock_ocr.call_count, 2)

    def test_small_synthetic_aadhaar_image_is_readable_after_upscaling(self):
        """Small fictional Aadhaar-style text should reach OCR at a readable size."""
        image = Image.new('RGB', (240, 140), 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 8)
        lines = [
            'GOVERNMENT OF INDIA',
            'AADHAAR CARD',
            'Name: KAVYA MEHTA',
            'Year of Birth: 1995',
            'Aadhaar Number: 1111 2222 3333',
        ]
        for index, line in enumerate(lines):
            draw.text((6, 12 + index * 20), line, fill='black', font=font)
        path = _save_test_image(image, self.test_dir, 'small_synthetic_aadhaar.png')

        if not self.engine.is_available()['available']:
            self.skipTest('Tesseract is not available')

        result = self.engine.extract_text(path)

        self.assertTrue(result['success'])
        self.assertIn('AADHAAR', result['text'].upper())
        self.assertIn('1111', result['text'])

    def test_extract_text_no_tesseract_gives_clear_error(self):
        """If Tesseract is not installed, the error message should be clear."""
        availability = self.engine.is_available()
        if not availability['available']:
            img = Image.new('RGB', (100, 100), 'white')
            path = _save_test_image(img, self.test_dir, 'no_tess.png')
            result = self.engine.extract_text(path)
            self.assertFalse(result['success'])
            self.assertIsNotNone(result['error'])
            # Error should mention Tesseract
            self.assertTrue(
                'tesseract' in result['error'].lower() or
                'not found' in result['error'].lower() or
                'not installed' in result['error'].lower(),
                f"Error should mention Tesseract: {result['error']}"
            )


# ============================================================
# Processing API Tests
# ============================================================

class TestProcessingAPI(unittest.TestCase):
    """Tests for POST /api/documents/<id>/process."""

    def setUp(self):
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
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _upload_test_image(self, text_lines=None, filename='test.png'):
        """Helper: upload a synthetic document image and return the document_id."""
        img = _create_test_image(text_lines)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        resp = self.client.post(
            '/api/documents/upload',
            data={'file': (buf, filename)},
            content_type='multipart/form-data',
        )
        return json.loads(resp.data)['document_id']

    def test_process_nonexistent_document(self):
        """Processing a nonexistent document should return 404."""
        resp = self.client.post('/api/documents/nonexistent-id/process')
        self.assertEqual(resp.status_code, 404)

    @patch('backend.services.document_services.OCREngine')
    def test_process_with_mocked_ocr(self, MockOCREngine):
        """Processing should work end-to-end with mocked OCR."""
        # Configure the mock OCR engine
        mock_engine = MockOCREngine.return_value
        mock_engine.extract_text.return_value = {
            "success": True,
            "text": "REPUBLIC OF INDIA\nPASSPORT\nSurname: SHARMA\nGiven Name: RAJESH\nDate of Birth: 15/03/1990\nPassport No: J8369854\nDate of Expiry: 14/03/2030",
            "error": None,
            "engine": "tesseract",
            "confidence": 85.0,
        }

        doc_id = self._upload_test_image()
        resp = self.client.post(f'/api/documents/{doc_id}/process')
        self.assertEqual(resp.status_code, 200)

        result = json.loads(resp.data)
        self.assertTrue(result['success'])
        self.assertEqual(result['document_id'], doc_id)
        self.assertEqual(result['processing_status'], 'processed')
        self.assertIn('document_type', result)
        self.assertIn('extracted_text', result)
        self.assertIn('extracted_fields', result)

    @patch('backend.services.document_services.OCREngine')
    def test_process_synthetic_aadhaar_image(self, MockOCREngine):
        """A fictional Aadhaar image should use the existing processing response."""
        MockOCREngine.return_value.extract_text.return_value = {
            "success": True,
            "text": (
                "Government of India\nAadhaar Card\n"
                "Name: KAVYA MEHTA\nYear of Birth: 1995\n"
                "Aadhaar Number: 9999 8888 7777"
            ),
            "error": None,
            "engine": "tesseract",
            "confidence": 92.0,
        }

        doc_id = self._upload_test_image(
            text_lines=[
                "Government of India",
                "Aadhaar Card",
                "Name: KAVYA MEHTA",
                "Year of Birth: 1995",
                "Aadhaar Number: 9999 8888 7777",
            ],
            filename='synthetic_aadhaar.png',
        )
        response = self.client.post(f'/api/documents/{doc_id}/process')
        result = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(result['document_type'], 'aadhaar')
        self.assertEqual(result['extracted_fields']['name'], '[REDACTED]')
        self.assertEqual(result['extracted_fields']['date_of_birth'], '[REDACTED]')
        self.assertEqual(result['extracted_fields']['document_number'], '[REDACTED]')
        self.assertIsNone(result['extracted_fields']['expiry_date'])

    @patch('backend.services.document_services.OCREngine')
    def test_process_returns_classification(self, MockOCREngine):
        """Processing should return a document type classification."""
        mock_engine = MockOCREngine.return_value
        mock_engine.extract_text.return_value = {
            "success": True,
            "text": "PASSPORT\nREPUBLIC\nSurname: DOE\nDate of Birth: 01/01/1990\nNationality: INDIAN",
            "error": None,
            "engine": "tesseract",
            "confidence": 80.0,
        }

        doc_id = self._upload_test_image()
        resp = self.client.post(f'/api/documents/{doc_id}/process')
        result = json.loads(resp.data)

        self.assertEqual(result['document_type'], 'passport')
        self.assertGreater(result['type_confidence'], 0.0)
        self.assertEqual(result['classification_method'], 'keyword_baseline')

    @patch('backend.services.document_services.OCREngine')
    def test_process_returns_redacted_fields(self, MockOCREngine):
        """Processing should return sanitized field data without exposing sensitive values."""
        mock_engine = MockOCREngine.return_value
        mock_engine.extract_text.return_value = {
            "success": True,
            "text": "Name: RAJESH SHARMA\nDate of Birth: 15/03/1990\nPassport No: J8369854\nExpiry Date: 14/03/2030",
            "error": None,
            "engine": "tesseract",
            "confidence": 90.0,
        }

        doc_id = self._upload_test_image()
        resp = self.client.post(f'/api/documents/{doc_id}/process')
        result = json.loads(resp.data)

        fields = result['extracted_fields']
        self.assertIn('name', fields)
        self.assertIn('date_of_birth', fields)
        self.assertIn('document_number', fields)
        self.assertIn('extraction_method', fields)
        self.assertEqual(fields['name'], '[REDACTED]')
        self.assertEqual(fields['date_of_birth'], '[REDACTED]')
        self.assertEqual(fields['document_number'], '[REDACTED]')
        self.assertEqual(fields['extraction_method'], 'regex_baseline')
        self.assertIsNone(result['extracted_text'])

    @patch('backend.services.document_services.OCREngine')
    def test_processing_result_does_not_persist_raw_ocr_text(self, MockOCREngine):
        """Persisted processing results should not keep the raw OCR text or sensitive field values."""
        mock_engine = MockOCREngine.return_value
        mock_engine.extract_text.return_value = {
            "success": True,
            "text": "Name: RAJESH SHARMA\nDate of Birth: 15/03/1990\nPassport No: J8369854",
            "error": None,
            "engine": "tesseract",
            "confidence": 90.0,
        }

        doc_id = self._upload_test_image()
        self.client.post(f'/api/documents/{doc_id}/process')

        result = self.client.get(f'/api/documents/{doc_id}/result')
        payload = json.loads(result.data)
        self.assertIsNone(payload['extracted_text'])
        self.assertEqual(payload['extracted_fields']['document_number'], '[REDACTED]')

    @patch('backend.services.document_services.OCREngine')
    def test_process_updates_status(self, MockOCREngine):
        """Processing should update document status to 'processed'."""
        mock_engine = MockOCREngine.return_value
        mock_engine.extract_text.return_value = {
            "success": True, "text": "PASSPORT", "error": None,
            "engine": "tesseract", "confidence": 50.0,
        }

        doc_id = self._upload_test_image()

        # Before processing: status should be 'uploaded'
        status_resp = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(json.loads(status_resp.data)['processing_status'], 'uploaded')

        # Process
        self.client.post(f'/api/documents/{doc_id}/process')

        # After processing: status should be 'processed'
        status_resp = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(json.loads(status_resp.data)['processing_status'], 'processed')

    @patch('backend.services.document_services.OCREngine')
    def test_process_ocr_failure(self, MockOCREngine):
        """OCR failure should set status to 'failed' and return error."""
        mock_engine = MockOCREngine.return_value
        mock_engine.extract_text.return_value = {
            "success": False, "text": None,
            "error": "Tesseract binary not found",
            "engine": "tesseract", "confidence": None,
        }

        doc_id = self._upload_test_image()
        resp = self.client.post(f'/api/documents/{doc_id}/process')
        self.assertEqual(resp.status_code, 422)

        result = json.loads(resp.data)
        self.assertFalse(result['success'])
        self.assertIn('error', result)

        # Status should be 'failed'
        status_resp = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(json.loads(status_resp.data)['processing_status'], 'failed')

    @patch('backend.services.document_services.OCREngine')
    def test_concurrent_processing_requests_only_process_once(self, MockOCREngine):
        """A second request should receive a conflict while the first is processing."""
        ocr_started = threading.Event()
        release_ocr = threading.Event()

        def extract_text_once(_file_path):
            ocr_started.set()
            self.assertTrue(release_ocr.wait(timeout=5))
            return {
                "success": True, "text": "PASSPORT", "error": None,
                "engine": "tesseract", "confidence": 80.0,
            }

        MockOCREngine.return_value.extract_text.side_effect = extract_text_once
        doc_id = self._upload_test_image()
        responses = {}

        def send_request(name):
            with self.app.test_client() as client:
                responses[name] = client.post(f'/api/documents/{doc_id}/process')

        first = threading.Thread(target=send_request, args=('first',))
        first.start()
        self.assertTrue(ocr_started.wait(timeout=5))

        second = threading.Thread(target=send_request, args=('second',))
        second.start()
        second.join(timeout=5)
        self.assertFalse(second.is_alive())
        self.assertEqual(responses['second'].status_code, 409)

        release_ocr.set()
        first.join(timeout=5)
        self.assertFalse(first.is_alive())
        self.assertEqual(responses['first'].status_code, 200)
        self.assertEqual(MockOCREngine.return_value.extract_text.call_count, 1)

        status = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(json.loads(status.data)['processing_status'], 'processed')

    @patch('backend.services.document_services.OCREngine')
    def test_unexpected_processing_failure_clears_processing_status(self, MockOCREngine):
        """An unexpected pipeline error should leave the document failed, not processing."""
        MockOCREngine.return_value.extract_text.side_effect = RuntimeError(
            "Unexpected OCR failure"
        )
        doc_id = self._upload_test_image()

        response = self.client.post(f'/api/documents/{doc_id}/process')
        self.assertEqual(response.status_code, 422)
        result = json.loads(response.data)
        self.assertFalse(result['success'])
        self.assertEqual(result['processing_status'], 'failed')
        self.assertEqual(result['error'], 'Document processing failed')
        self.assertNotIn('Unexpected OCR failure', response.data.decode())

        status = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(json.loads(status.data)['processing_status'], 'failed')

    @patch('backend.services.document_services.OCREngine')
    def test_process_empty_ocr_text_fails(self, MockOCREngine):
        """Successful OCR with empty text should not be marked processed."""
        MockOCREngine.return_value.extract_text.return_value = {
            "success": True, "text": "", "error": None,
            "engine": "tesseract", "confidence": None,
        }

        doc_id = self._upload_test_image()
        resp = self.client.post(f'/api/documents/{doc_id}/process')

        self.assertEqual(resp.status_code, 422)
        result = json.loads(resp.data)
        self.assertFalse(result['success'])
        self.assertEqual(result['processing_status'], 'failed')
        self.assertEqual(result['document_type'], 'unknown')
        self.assertIn('no readable text', result['error'])

        status = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(json.loads(status.data)['processing_status'], 'failed')

    @patch('backend.services.document_services.OCREngine')
    def test_process_whitespace_ocr_text_fails(self, MockOCREngine):
        """Successful OCR with whitespace-only text should not be marked processed."""
        MockOCREngine.return_value.extract_text.return_value = {
            "success": True, "text": " \n\t ", "error": None,
            "engine": "tesseract", "confidence": None,
        }

        doc_id = self._upload_test_image()
        resp = self.client.post(f'/api/documents/{doc_id}/process')

        self.assertEqual(resp.status_code, 422)
        self.assertFalse(json.loads(resp.data)['success'])
        status = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(json.loads(status.data)['processing_status'], 'failed')

    @patch('backend.services.document_services.DocumentClassifier')
    @patch('backend.services.document_services.OCREngine')
    def test_process_unknown_classification_fails(self, MockOCREngine, MockClassifier):
        """Unknown classification should preserve output and fail processing."""
        MockOCREngine.return_value.extract_text.return_value = {
            "success": True, "text": "unrecognized document text", "error": None,
            "engine": "tesseract", "confidence": 60.0,
        }
        MockClassifier.return_value.classify.return_value = {
            "document_type": "unknown", "confidence": 0.0,
            "method": "keyword_baseline", "all_scores": {},
        }

        doc_id = self._upload_test_image()
        resp = self.client.post(f'/api/documents/{doc_id}/process')

        self.assertEqual(resp.status_code, 422)
        result = json.loads(resp.data)
        self.assertFalse(result['success'])
        self.assertEqual(result['document_type'], 'unknown')
        self.assertEqual(result['type_confidence'], 0.0)
        self.assertIn('unknown', result['error'])
        status = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(json.loads(status.data)['processing_status'], 'failed')

    @patch('backend.services.document_services.DocumentClassifier')
    @patch('backend.services.document_services.OCREngine')
    def test_process_low_confidence_classification_fails(self, MockOCREngine, MockClassifier):
        """Classification below the threshold should not be marked processed."""
        MockOCREngine.return_value.extract_text.return_value = {
            "success": True, "text": "possible passport text", "error": None,
            "engine": "tesseract", "confidence": 60.0,
        }
        MockClassifier.return_value.classify.return_value = {
            "document_type": "passport", "confidence": 0.1,
            "method": "keyword_baseline", "all_scores": {"passport": 0.1},
        }

        doc_id = self._upload_test_image()
        resp = self.client.post(f'/api/documents/{doc_id}/process')

        self.assertEqual(resp.status_code, 422)
        result = json.loads(resp.data)
        self.assertFalse(result['success'])
        self.assertEqual(result['document_type'], 'passport')
        self.assertEqual(result['type_confidence'], 0.1)
        self.assertIn('below the minimum threshold', result['error'])
        status = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(json.loads(status.data)['processing_status'], 'failed')

    @patch('backend.services.document_services.OCREngine')
    def test_process_missing_file_on_disk(self, MockOCREngine):
        """Processing a document whose file was deleted should fail gracefully."""
        doc_id = self._upload_test_image()

        # Delete the uploaded file from disk
        for f in os.listdir(self.upload_folder):
            if f.startswith(doc_id):
                os.remove(os.path.join(self.upload_folder, f))

        resp = self.client.post(f'/api/documents/{doc_id}/process')
        result = json.loads(resp.data)
        self.assertFalse(result['success'])
        self.assertIn('not found', result['error'].lower())

    @patch('backend.services.document_services.OCREngine')
    def test_existing_upload_endpoints_still_work(self, MockOCREngine):
        """Phase 1 upload and status endpoints must remain functional."""
        # Upload
        img = _create_test_image()
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        resp = self.client.post(
            '/api/documents/upload',
            data={'file': (buf, 'compat_test.png')},
            content_type='multipart/form-data',
        )
        self.assertEqual(resp.status_code, 200)
        doc_id = json.loads(resp.data)['document_id']

        # Status
        status_resp = self.client.get(f'/api/documents/{doc_id}/status')
        self.assertEqual(status_resp.status_code, 200)

        # Health
        health_resp = self.client.get('/api/health')
        self.assertEqual(health_resp.status_code, 200)


class TestProcessingResultEndpoint(unittest.TestCase):
    """Tests for GET /api/documents/<id>/result."""

    def setUp(self):
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
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_result_before_processing(self):
        """Requesting results for an unprocessed document should return 404."""
        img = _create_test_image()
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        resp = self.client.post(
            '/api/documents/upload',
            data={'file': (buf, 'no_result.png')},
            content_type='multipart/form-data',
        )
        doc_id = json.loads(resp.data)['document_id']

        result_resp = self.client.get(f'/api/documents/{doc_id}/result')
        self.assertEqual(result_resp.status_code, 404)

    @patch('backend.services.document_services.OCREngine')
    def test_result_after_processing(self, MockOCREngine):
        """After processing, results should be retrievable."""
        mock_engine = MockOCREngine.return_value
        mock_engine.extract_text.return_value = {
            "success": True,
            "text": "PASSPORT\nName: DOE\nDate of Birth: 01/01/1990",
            "error": None, "engine": "tesseract", "confidence": 75.0,
        }

        img = _create_test_image()
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        resp = self.client.post(
            '/api/documents/upload',
            data={'file': (buf, 'with_result.png')},
            content_type='multipart/form-data',
        )
        doc_id = json.loads(resp.data)['document_id']

        # Process the document
        self.client.post(f'/api/documents/{doc_id}/process')

        # Retrieve results
        result_resp = self.client.get(f'/api/documents/{doc_id}/result')
        self.assertEqual(result_resp.status_code, 200)
        result = json.loads(result_resp.data)
        self.assertEqual(result['document_id'], doc_id)
        self.assertIn('document_type', result)
        self.assertIn('extracted_fields', result)
        self.assertIn('processing_status', result)

    def test_result_nonexistent_document(self):
        """Results for a nonexistent document should return 404."""
        resp = self.client.get('/api/documents/fake-id/result')
        self.assertEqual(resp.status_code, 404)


class TestAadhaarRegressionSuite(unittest.TestCase):
    """Focused regression tests for Aadhaar field extraction, layouts, and pipeline."""

    def setUp(self):
        self.extractor = FieldExtractor()
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        self.upload_folder = os.path.join(self.test_dir, 'uploads')
        self.app = create_app({
            'TESTING': True,
            'DATABASE_PATH': self.db_path,
            'UPLOAD_FOLDER': self.upload_folder,
            'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'bmp', 'tiff'},
        })
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_extract_unlabeled_standard_aadhaar_fields(self):
        """A standard Aadhaar card without 'Name:' label should extract name and compound DOB."""
        text = (
            "Government of India\n"
            "Unique Identification Authority of India\n"
            "Enrolment No.: 1234/56789/01234\n"
            "Kavya Mehta\n"
            "Your Aadhaar No\n"
            "Date of Birth/DOB: 15/08/1995\n"
            "Male/ MALE\n"
            "1111 2222 3333"
        )
        fields = self.extractor.extract_fields(text, document_type='aadhaar')
        self.assertEqual(fields['name'], 'Kavya Mehta')
        self.assertEqual(fields['date_of_birth'], '15/08/1995')
        self.assertEqual(fields['document_number'], '1111 2222 3333')
        self.assertIsNone(fields['expiry_date'])
        self.assertEqual(fields['field_count'], 3)

    def test_extract_aadhaar_letter_format_name(self):
        """An Aadhaar letter format with To\\n<Name>\\nS/O should extract the name."""
        text = (
            "Government of India\n"
            "Aadhaar\n"
            "To\n"
            "Arjun Sharma\n"
            "S/O Ramesh Sharma\n"
            "PO: Central City\n"
            "Your Aadhaar No. : 1111 2222 3333\n"
            "Date of Birth: 08/04/1987"
        )
        fields = self.extractor.extract_fields(text, document_type='aadhaar')
        self.assertEqual(fields['name'], 'Arjun Sharma')
        self.assertEqual(fields['date_of_birth'], '08/04/1987')
        self.assertEqual(fields['document_number'], '1111 2222 3333')

    def test_bilingual_dob_pattern_matching(self):
        """Bilingual DOB patterns like 'DOB / जन्म तारीख: DD/MM/YYYY' should extract clean date."""
        for text, expected in [
            ("Date of Birth/DOB: 03/03/2008", "03/03/2008"),
            ("DOB: 15/03/1990", "15/03/1990"),
            ("DOB / जन्म तारीख: 25-12-1985", "25-12-1985"),
            ("Year of Birth: 1995", "1995"),
        ]:
            with self.subTest(text=text):
                fields = self.extractor.extract_fields(text, document_type='aadhaar')
                self.assertEqual(fields['date_of_birth'], expected)

    @patch('backend.services.document_services.OCREngine')
    def test_aadhaar_pipeline_synthetic_identity_matching(self, MockOCREngine):
        """An uploaded synthetic Aadhaar image should classify, extract 3 fields, and match synthetic record."""
        MockOCREngine.return_value.extract_text.return_value = {
            "success": True,
            "text": (
                "Government of India\n"
                "Unique Identification Authority of India\n"
                "Enrolment No.: 1234/56789/01234\n"
                "KAVYA MEHTA\n"
                "Date of Birth/DOB: 1995\n"
                "1111 2222 3333"
            ),
            "error": None,
            "engine": "tesseract",
            "confidence": 92.0,
        }

        img = Image.new('RGB', (400, 300), 'white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        upload_resp = self.client.post(
            '/api/documents/upload',
            data={'file': (buf, 'synthetic_aadhaar.png')},
            content_type='multipart/form-data',
        )
        doc_id = json.loads(upload_resp.data)['document_id']

        proc_resp = self.client.post(f'/api/documents/{doc_id}/process')
        self.assertEqual(proc_resp.status_code, 200)
        res = json.loads(proc_resp.data)

        self.assertEqual(res['document_type'], 'aadhaar')
        self.assertEqual(res['extracted_fields']['name'], '[REDACTED]')
        self.assertEqual(res['extracted_fields']['date_of_birth'], '[REDACTED]')
        self.assertEqual(res['extracted_fields']['document_number'], '[REDACTED]')
        self.assertEqual(res['extracted_fields']['field_count'], 3)
        self.assertEqual(res['identity_match']['status'], 'Match')
        self.assertEqual(res['identity_match']['record_id'], 'SYN-AADHAAR-001')


if __name__ == '__main__':
    unittest.main()

