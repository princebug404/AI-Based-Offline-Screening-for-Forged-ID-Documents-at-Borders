"""Privacy-conscious face comparison for one document and one live image."""

import cv2
import numpy as np

from src.face.face_detector import FaceDetector


class FaceVerifier:
	"""Compare exactly one detected portrait in each image using cosine similarity."""

	def __init__(self, detector=None, default_threshold=0.45):
		self.detector = detector or FaceDetector()
		self.default_threshold = default_threshold

	def compare(self, document_source, live_image, threshold=None):
		decision_threshold = (
			self.default_threshold if threshold is None else float(threshold)
		)
		base = {
			'comparison_status': 'Manual Review',
			'similarity_score': None,
			'decision_threshold': decision_threshold,
			'document_face_count': None,
			'live_face_count': None,
			'embedding_model': self.detector.model_name,
			'similarity_metric': 'cosine',
		}

		try:
			document_image, document_faces = self.detector.detect_document_portrait(
				document_source
			)
			live_image_array, live_faces = self.detector.detect(live_image)
			base['document_face_count'] = len(document_faces)
			base['live_face_count'] = len(live_faces)

			if len(document_faces) != 1 or len(live_faces) != 1:
				base['reason'] = 'Exactly one usable face is required in each image.'
				return base

			document_face = document_faces[0]
			live_face = live_faces[0]
			if not self._usable_face(document_image, document_face):
				base['reason'] = 'The document portrait is too small, blurry, or low quality.'
				return base
			if not self._usable_face(live_image_array, live_face):
				base['reason'] = 'The live face image is too small, blurry, or low quality.'
				return base

			document_embedding = self._normalized_embedding(document_face)
			live_embedding = self._normalized_embedding(live_face)
			similarity = float(np.dot(document_embedding, live_embedding))
			base['similarity_score'] = round(similarity, 4)
			base['comparison_status'] = (
				'Match' if similarity >= decision_threshold else 'No Match'
			)
			return base
		except Exception as error:
			base['reason'] = f'Face comparison unavailable: {error}'
			return base

	@staticmethod
	def _normalized_embedding(face):
		embedding = np.asarray(face.normed_embedding, dtype=np.float32)
		norm = np.linalg.norm(embedding)
		if embedding.ndim != 1 or norm == 0 or not np.isfinite(norm):
			raise ValueError('Invalid face embedding')
		return embedding / norm

	@staticmethod
	def _usable_face(image, face, min_size=50):
		if float(getattr(face, 'det_score', 0.0)) < 0.5:
			return False

		x1, y1, x2, y2 = face.bbox.astype(int)
		x1, y1 = max(0, x1), max(0, y1)
		x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
		if x2 - x1 < min_size or y2 - y1 < min_size:
			return False

		crop = image[y1:y2, x1:x2]
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= 20.0
