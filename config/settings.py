import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    format: str = "int16"
    device_index: Optional[int] = None


@dataclass
class ModelConfig:
    whisper_model: str = "medium"  # tiny, base, small, medium, large
    llama_model: str = "llama-2-7b-chat"
    device: str = "cuda" if os.name != "nt" else "cpu"  # For Iran, might use CPU
    max_memory: str = "8GB"
    quantization: bool = True


@dataclass
class AnalysisConfig:
    sentiment_threshold: float = 0.5
    profanity_threshold: float = 0.7
    stress_threshold: float = 0.6
    emotion_classes: List[str] = None

    def __post_init__(self):
        if self.emotion_classes is None:
            self.emotion_classes = [
                "anger",
                "joy",
                "sadness",
                "fear",
                "surprise",
                "disgust",
                "neutral",
            ]


@dataclass
class DatabaseConfig:
    db_path: str = "data/call_analysis.db"
    backup_interval: int = 3600  # seconds


@dataclass
class UIConfig:
    theme: str = "dark"
    language: str = "en"  # Can be changed to "fa" for Persian
    window_size: tuple = (1200, 800)
    update_interval: int = 100  # milliseconds


class Settings:
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.LOGS_DIR = self.DATA_DIR / "logs"

        # Create directories
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            dir_path.mkdir(exist_ok=True)

        self.audio = AudioConfig()
        self.model = ModelConfig()
        self.analysis = AnalysisConfig()
        self.database = DatabaseConfig()
        self.ui = UIConfig()

        # Persian profanity words (example list)
        self.PERSIAN_PROFANITY = [
            "کسکش",
            "جنده",
            "گوه",
            "کیری",
            "کونی",
            "آشغال",
            # Add more as needed
        ]

        # English profanity words
        self.ENGLISH_PROFANITY = [
            "fuck",
            "shit",
            "damn",
            "bitch",
            "asshole",
            "bastard",
            # Add more as needed
        ]

    def get_model_path(self, model_name: str) -> Path:
        return self.MODELS_DIR / model_name

    def get_data_path(self, filename: str) -> Path:
        return self.DATA_DIR / filename


# Global settings instance
settings = Settings()
