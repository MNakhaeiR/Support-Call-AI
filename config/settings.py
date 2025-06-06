# Configuration settings for the application

API_KEY = "your_api_key_here"
DATABASE_URL = "sqlite:///your_database.db"
LOG_LEVEL = "INFO"

# Model configurations
WHISPER_MODEL_PATH = "models/whisper/model.pt"
LLAMA_MODEL_PATH = "models/llama/model.pt"

# Other environment-specific settings
DEBUG_MODE = True
MAX_AUDIO_DURATION = 60  # in seconds
SUPPORTED_AUDIO_FORMATS = ["wav", "mp3", "flac"]