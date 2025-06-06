from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit

class AnalysisDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.title_label = QLabel("Analysis Results")
        self.results_text_edit = QTextEdit()
        self.results_text_edit.setReadOnly(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.results_text_edit)

        self.setLayout(layout)

    def update_results(self, results):
        self.results_text_edit.setPlainText(results)