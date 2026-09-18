from PySide6.QtWidgets import QMainWindow, QStackedWidget

from desktop.api_client import ApiClient
from desktop.ui.capture_page import CapturePage
from desktop.ui.results_page import ResultsPage


class MainWindow(QMainWindow):
	"""Minimal desktop shell for the existing upload/process API workflow."""

	def __init__(self, base_url="http://127.0.0.1:5055"):
		super().__init__()
		self.setWindowTitle("SIH26188 Document Screening")
		self.resize(640, 420)

		self.pages = QStackedWidget()
		self.results_page = ResultsPage(self.show_capture_page)
		self.capture_page = CapturePage(ApiClient(base_url), self.show_result)
		self.pages.addWidget(self.capture_page)
		self.pages.addWidget(self.results_page)
		self.setCentralWidget(self.pages)

	def show_result(self, result):
		self.results_page.show_result(result)
		self.pages.setCurrentWidget(self.results_page)

	def show_capture_page(self):
		self.pages.setCurrentWidget(self.capture_page)
