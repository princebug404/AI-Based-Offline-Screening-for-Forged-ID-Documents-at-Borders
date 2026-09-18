import os
import sys

from PySide6.QtWidgets import QApplication

from desktop.ui.main_window import MainWindow


def main():
	app = QApplication(sys.argv)
	base_url = os.environ.get('SIH_API_URL', 'http://127.0.0.1:5055')
	window = MainWindow(base_url)
	window.show()
	return app.exec()


if __name__ == '__main__':
	raise SystemExit(main())
