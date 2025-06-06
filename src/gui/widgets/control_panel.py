from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QSlider

class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super(ControlPanel, self).__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("Control Panel", self)
        layout.addWidget(self.label)

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_action)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_action)
        layout.addWidget(self.stop_button)

        self.volume_slider = QSlider(self)
        self.volume_slider.setOrientation(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        layout.addWidget(self.volume_slider)

        self.setLayout(layout)

    def start_action(self):
        # Logic to start the process
        print("Started")

    def stop_action(self):
        # Logic to stop the process
        print("Stopped")