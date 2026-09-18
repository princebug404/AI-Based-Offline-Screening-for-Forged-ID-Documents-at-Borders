import json

from PySide6.QtWidgets import QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


class ResultsPage(QWidget):
	"""Display the existing processing response without changing its data."""

	def __init__(self, on_back, parent=None):
		super().__init__(parent)
		self.status_label = QLabel("No result yet")
		self.result_text = QTextEdit()
		self.result_text.setReadOnly(True)
		back_button = QPushButton("Process another image")
		back_button.clicked.connect(on_back)

		layout = QVBoxLayout(self)
		layout.addWidget(QLabel("Processing result"))
		layout.addWidget(self.status_label)
		layout.addWidget(self.result_text)
		layout.addWidget(back_button)

	def show_result(self, result):
		status = result.get('processing_status', 'unknown')
		self.status_label.setText(f"Status: {status}")
		self.result_text.setPlainText(json.dumps(result, indent=2, sort_keys=True))
