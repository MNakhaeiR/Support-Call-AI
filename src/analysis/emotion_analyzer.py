from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import numpy as np

class EmotionAnalyzer:
    def __init__(self, model_path='models/emotion_model.pkl'):
        self.vectorizer = CountVectorizer()
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        try:
            return joblib.load(model_path)
        except FileNotFoundError:
            raise Exception("Model file not found. Please train the model first.")

    def preprocess_text(self, text):
        return [text]

    def predict_emotion(self, text):
        processed_text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform(processed_text)
        prediction = self.model.predict(text_vector)
        return prediction[0]

    def train_model(self, training_data, labels):
        training_vectors = self.vectorizer.fit_transform(training_data)
        self.model = MultinomialNB()
        self.model.fit(training_vectors, labels)
        joblib.dump(self.model, 'models/emotion_model.pkl')

    def evaluate_model(self, test_data, test_labels):
        test_vectors = self.vectorizer.transform(test_data)
        accuracy = self.model.score(test_vectors, test_labels)
        return accuracy
