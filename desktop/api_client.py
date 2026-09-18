import requests


class ApiClient:
	"""Small client for the existing document-processing API."""

	def __init__(self, base_url="http://127.0.0.1:5055", timeout=15):
		self.base_url = base_url.rstrip('/')
		self.timeout = timeout

	def upload_document(self, file_path):
		with open(file_path, 'rb') as document_file:
			response = requests.post(
				f"{self.base_url}/api/documents/upload",
				files={'file': (file_path.name, document_file)},
				timeout=self.timeout,
			)
		return self._response_data(response)

	def process_document(self, document_id):
		response = requests.post(
			f"{self.base_url}/api/documents/{document_id}/process",
			timeout=self.timeout,
		)
		return self._response_data(response, raise_for_status=False)

	def get_result(self, document_id):
		response = requests.get(
			f"{self.base_url}/api/documents/{document_id}/result",
			timeout=self.timeout,
		)
		return self._response_data(response)

	@staticmethod
	def _response_data(response, raise_for_status=True):
		try:
			data = response.json()
		except ValueError:
			data = {'error': response.text or 'Backend returned an invalid response'}

		if response.ok or not raise_for_status:
			return data

		error = data.get('error', f'Backend request failed ({response.status_code})')
		raise RuntimeError(error)
