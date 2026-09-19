import io
import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from backend.app import create_app
from src.face.face_verfier import FaceVerifier


class StubDetector:
	"""Deterministic detector for comparison behavior tests."""

	model_name = 'stub-embedding-model'

	def __init__(self, faces):
		self.faces = faces

	def detect(self, _source):
		image = np.random.default_rng(7).integers(
			0, 256, size=(240, 240, 3), dtype=np.uint8
		)
		return image, self.faces

	def detect_document_portrait(self, source):
		return self.detect(source)


def _face(embedding, score=0.99):
	return SimpleNamespace(
		bbox=np.array([40, 40, 200, 200], dtype=np.float32),
		det_score=score,
		normed_embedding=np.array(embedding, dtype=np.float32),
	)


class TestFaceVerifier(unittest.TestCase):
	def test_match_returns_cosine_score_and_threshold(self):
		verifier = FaceVerifier(
			detector=StubDetector([_face([1.0, 0.0])]),
			default_threshold=0.8,
		)

		result = verifier.compare('document', b'live')

		self.assertEqual(result['comparison_status'], 'Match')
		self.assertEqual(result['similarity_score'], 1.0)
		self.assertEqual(result['decision_threshold'], 0.8)
		self.assertEqual(result['similarity_metric'], 'cosine')
		self.assertEqual(result['embedding_model'], 'stub-embedding-model')

	def test_document_portrait_extraction_is_used(self):

		class PortraitOnlyDetector(StubDetector):
			def __init__(self):
				super().__init__([_face([1.0, 0.0])])
				self.portrait_calls = 0

			def detect_document_portrait(self, source):
				self.portrait_calls += 1
				return super().detect_document_portrait(source)

		detector = PortraitOnlyDetector()
		result = FaceVerifier(detector=detector).compare('document', b'live')

		self.assertEqual(result['comparison_status'], 'Match')
		self.assertEqual(detector.portrait_calls, 1)

	def test_different_faces_return_no_match(self):
		class DifferentDetector(StubDetector):
			def detect(self, source):
				self.faces = [_face([1.0, 0.0]) if source == 'document' else _face([0.0, 1.0])]
				return super().detect(source)

		result = FaceVerifier(
			detector=DifferentDetector([]),
			default_threshold=0.8,
		).compare('document', b'live')

		self.assertEqual(result['comparison_status'], 'No Match')
		self.assertEqual(result['similarity_score'], 0.0)

	def test_missing_or_multiple_faces_require_manual_review(self):
		for faces in ([], [_face([1.0, 0.0]), _face([1.0, 0.0])]):
			with self.subTest(face_count=len(faces)):
				result = FaceVerifier(detector=StubDetector(faces)).compare(
					'document', b'live'
				)
				self.assertEqual(result['comparison_status'], 'Manual Review')
				self.assertIsNone(result['similarity_score'])

	@patch.object(FaceVerifier, '_usable_face', return_value=False)
	def test_low_quality_face_requires_manual_review(self, _quality_check):
		result = FaceVerifier(detector=StubDetector([_face([1.0, 0.0])])).compare(
			'document', b'live'
		)
		self.assertEqual(result['comparison_status'], 'Manual Review')
		self.assertIn('low quality', result['reason'])

	def test_document_portrait_usable_at_standard_id_size(self):
		"""A standard ID document portrait (e.g. 65x65 px) should be usable."""
		face = _face([1.0, 0.0], score=0.85)
		face.bbox = np.array([10, 10, 75, 75], dtype=np.float32)  # 65x65 px
		image = np.random.default_rng(42).integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
		self.assertTrue(FaceVerifier._usable_face(image, face, min_size=50))

	def test_face_below_minimum_size_rejected(self):
		"""A portrait smaller than minimum size (e.g. 35x35 px) should be rejected."""
		face = _face([1.0, 0.0], score=0.85)
		face.bbox = np.array([10, 10, 45, 45], dtype=np.float32)  # 35x35 px
		image = np.random.default_rng(42).integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
		self.assertFalse(FaceVerifier._usable_face(image, face, min_size=50))


class TestFaceComparisonRoute(unittest.TestCase):
	def setUp(self):
		self.test_dir = tempfile.mkdtemp()
		self.app = create_app({
			'TESTING': True,
			'DATABASE_PATH': os.path.join(self.test_dir, 'test.db'),
			'UPLOAD_FOLDER': os.path.join(self.test_dir, 'uploads'),
			'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'bmp', 'tiff'},
			'FACE_MATCH_THRESHOLD': 0.8,
		})
		self.client = self.app.test_client()

	def tearDown(self):
		shutil.rmtree(self.test_dir, ignore_errors=True)

	def test_face_image_is_sent_to_comparison_without_being_saved(self):
		document = Image.new('RGB', (120, 120), 'white')
		document_buffer = io.BytesIO()
		document.save(document_buffer, format='PNG')
		document_buffer.seek(0)
		upload = self.client.post(
			'/api/documents/upload',
			data={'file': (document_buffer, 'synthetic-document.png')},
			content_type='multipart/form-data',
		)
		document_id = json.loads(upload.data)['document_id']

		comparison = {
			'comparison_status': 'Manual Review',
			'similarity_score': None,
			'decision_threshold': 0.8,
		}
		with patch('backend.routes.document_routes._face_verifier') as verifier:
			verifier.compare.return_value = comparison
			response = self.client.post(
				f'/api/documents/{document_id}/compare-face',
				data={'face': (io.BytesIO(b'face-image-bytes'), 'live-face.png')},
				content_type='multipart/form-data',
			)

		self.assertEqual(response.status_code, 200)
		self.assertEqual(json.loads(response.data)['comparison_status'], 'Manual Review')
		verifier.compare.assert_called_once()
		self.assertEqual(len(os.listdir(self.app.config['UPLOAD_FOLDER'])), 1)

	def test_missing_live_face_is_rejected(self):
		response = self.client.post('/api/documents/missing/compare-face')
		self.assertEqual(response.status_code, 400)
		self.assertIn('live face image', json.loads(response.data)['error'].lower())

	def test_unsupported_face_image_type_is_rejected(self):
		response = self.client.post(
			'/api/documents/missing/compare-face',
			data={'face': (io.BytesIO(b'not a face'), 'live-face.pdf')},
			content_type='multipart/form-data',
		)
		self.assertEqual(response.status_code, 400)
		self.assertIn('not allowed', json.loads(response.data)['error'].lower())

	def test_invalid_face_image_returns_manual_review_without_persistence(self):
		image = Image.new('RGB', (120, 120), 'white')
		document_buffer = io.BytesIO()
		image.save(document_buffer, format='PNG')
		document_buffer.seek(0)
		upload = self.client.post(
			'/api/documents/upload',
			data={'file': (document_buffer, 'synthetic-document.png')},
			content_type='multipart/form-data',
		)
		document_id = json.loads(upload.data)['document_id']

		response = self.client.post(
			f'/api/documents/{document_id}/compare-face',
			data={'face': (io.BytesIO(b'not an image'), 'live-face.png')},
			content_type='multipart/form-data',
		)
		payload = json.loads(response.data)
		self.assertEqual(response.status_code, 200)
		self.assertEqual(payload['comparison_status'], 'Manual Review')
		self.assertIsNone(payload['similarity_score'])
		self.assertEqual(len(os.listdir(self.app.config['UPLOAD_FOLDER'])), 1)
