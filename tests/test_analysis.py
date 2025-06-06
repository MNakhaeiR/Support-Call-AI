import unittest
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.profanity_detector import ProfanityDetector
from src.analysis.emotion_analyzer import EmotionAnalyzer
from src.analysis.stress_detector import StressDetector
from src.analysis.llm_analyzer import LLMAnalyzer

class TestAnalysisFunctions(unittest.TestCase):

    def setUp(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.profanity_detector = ProfanityDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.stress_detector = StressDetector()
        self.llm_analyzer = LLMAnalyzer()

    def test_sentiment_analysis(self):
        text = "I love programming!"
        result = self.sentiment_analyzer.analyze(text)
        self.assertIn(result, ['positive', 'negative', 'neutral'])

    def test_profanity_detection(self):
        text = "This is a damn test."
        result = self.profanity_detector.detect(text)
        self.assertTrue(result)

    def test_emotion_analysis(self):
        text = "I am feeling very happy today!"
        result = self.emotion_analyzer.analyze(text)
        self.assertIn(result, ['happy', 'sad', 'angry', 'surprised'])

    def test_stress_detection(self):
        audio_file = "path/to/audio/file.wav"
        result = self.stress_detector.detect(audio_file)
        self.assertIsInstance(result, float)  # Assuming it returns a stress level as a float

    def test_llm_analysis(self):
        text = "What is the capital of France?"
        result = self.llm_analyzer.analyze(text)
        self.assertIsInstance(result, str)  # Assuming it returns a string response

if __name__ == '__main__':
    unittest.main()