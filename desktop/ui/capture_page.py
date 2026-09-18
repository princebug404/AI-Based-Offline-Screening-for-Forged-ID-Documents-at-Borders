from pathlib import Path

from PySide6.QtWidgets import (
	QFileDialog,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QVBoxLayout,
	QWidget,
)


class CapturePage(QWidget):
	"""Upload a supported document image and start processing."""

	def __init__(self, api_client, on_result, parent=None):
		super().__init__(parent)
		self.api_client = api_client
		self.on_result = on_result
		self.selected_path = None

		self.file_label = QLabel("No image selected")
		self.status_label = QLabel("Choose a PNG, JPG, JPEG, BMP, or TIFF image.")
		self.choose_button = QPushButton("Choose image")
		self.process_button = QPushButton("Upload and process")
		self.process_button.setEnabled(False)

		button_row = QHBoxLayout()
		button_row.addWidget(self.choose_button)
		button_row.addWidget(self.process_button)

		layout = QVBoxLayout(self)
		layout.addWidget(QLabel("Document screening"))
		layout.addWidget(self.file_label)
		layout.addLayout(button_row)
		layout.addWidget(self.status_label)
		layout.addStretch()

		self.choose_button.clicked.connect(self.choose_file)
		self.process_button.clicked.connect(self.upload_and_process)

	def choose_file(self):
		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Choose document image",
			"",
			"Images (*.png *.jpg *.jpeg *.bmp *.tiff)",
		)
		if not file_path:
			return

		self.selected_path = Path(file_path)
		self.file_label.setText(self.selected_path.name)
		self.status_label.setText("Ready to upload.")
		self.process_button.setEnabled(True)

	def upload_and_process(self):
		if self.selected_path is None:
			return

		self.choose_button.setEnabled(False)
		self.process_button.setEnabled(False)
		self.status_label.setText("Uploading document...")
		try:
			upload = self.api_client.upload_document(self.selected_path)
			document_id = upload['document_id']
			self.status_label.setText("Processing document...")
			result = self.api_client.process_document(document_id)
			result.setdefault('audit', upload.get('audit'))
			self.on_result(result)
		except Exception as error:
			self.on_result({
				'success': False,
				'processing_status': 'failed',
				'error': str(error),
			})
		finally:
			self.choose_button.setEnabled(True)
			self.process_button.setEnabled(self.selected_path is not None)
