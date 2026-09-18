"""Face detection backed by the pretrained InsightFace model pack."""

import cv2
import numpy as np


class FaceDetector:
	"""Lazily load InsightFace and detect faces without storing image data."""

	def __init__(self, model_name='buffalo_l', providers=None, det_size=(640, 640)):
		self.model_name = model_name
		self.providers = providers or ['CPUExecutionProvider']
		self.det_size = det_size
		self._model = None

	def _get_model(self):
		if self._model is None:
			from insightface.app import FaceAnalysis

			self._model = FaceAnalysis(
				name=self.model_name,
				providers=self.providers,
			)
			self._model.prepare(ctx_id=0, det_size=self.det_size)
		return self._model

	@staticmethod
	def decode_image(source):
		if isinstance(source, (bytes, bytearray, memoryview)):
			buffer = np.frombuffer(source, dtype=np.uint8)
			image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
		elif isinstance(source, str):
			image = cv2.imread(source, cv2.IMREAD_COLOR)
		else:
			image = source

		if image is None or not isinstance(image, np.ndarray) or image.size == 0:
			raise ValueError('Unable to decode image')
		return image

	def detect(self, source):
		image = self.decode_image(source)
		return image, self._get_model().get(image)
