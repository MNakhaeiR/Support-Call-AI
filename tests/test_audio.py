import unittest
from src.audio.audio_capture import AudioCapture
from src.audio.audio_preprocessor import AudioPreprocessor
from src.audio.speech_to_text import SpeechToText

class TestAudioProcessing(unittest.TestCase):

    def setUp(self):
        self.audio_capture = AudioCapture()
        self.audio_preprocessor = AudioPreprocessor()
        self.speech_to_text = SpeechToText()

    def test_audio_capture(self):
        audio_data = self.audio_capture.capture()
        self.assertIsNotNone(audio_data)
        self.assertGreater(len(audio_data), 0)

    def test_audio_preprocessing(self):
        raw_audio = self.audio_capture.capture()
        processed_audio = self.audio_preprocessor.preprocess(raw_audio)
        self.assertIsNotNone(processed_audio)
        self.assertNotEqual(raw_audio, processed_audio)

    def test_speech_to_text_conversion(self):
        audio_data = self.audio_capture.capture()
        processed_audio = self.audio_preprocessor.preprocess(audio_data)
        text = self.speech_to_text.convert(processed_audio)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

if __name__ == '__main__':
    unittest.main()