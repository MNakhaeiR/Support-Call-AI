import numpy as np
import librosa

def load_audio(file_path, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def reduce_noise(audio, noise_factor=0.1):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def preprocess_audio(file_path):
    audio = load_audio(file_path)
    audio = normalize_audio(audio)
    audio = reduce_noise(audio)
    return audio