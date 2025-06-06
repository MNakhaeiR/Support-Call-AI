# main.py

import sys
from src.audio.audio_capture import AudioCapture
from src.audio.audio_preprocessor import AudioPreprocessor
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.gui.main_window import MainWindow
from src.utils.logger import setup_logger

def main():
    # Set up logging
    logger = setup_logger()

    # Initialize components
    audio_capture = AudioCapture()
    audio_preprocessor = AudioPreprocessor()
    sentiment_analyzer = SentimentAnalyzer()
    main_window = MainWindow()

    # Start the main application loop
    main_window.run()

if __name__ == "__main__":
    main()