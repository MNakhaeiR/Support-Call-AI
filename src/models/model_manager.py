import os
import json
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from urllib.parse import urlparse
import time
import threading
import requests
from loguru import logger

# Import model libraries with error handling
try:
    import whisper

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    logger.warning("Whisper not available. Install with: pip install openai-whisper")

try:
    from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    logger.warning(
        "Hugging Face Hub not available. Install with: pip install huggingface_hub"
    )

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning(
        "Transformers not available. Install with: pip install transformers torch"
    )

try:
    from llama_cpp import Llama

    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    logger.warning(
        "llama-cpp-python not available. Install with: pip install llama-cpp-python"
    )

from config.settings import settings


class ModelManager:
    """Advanced model management system for downloading, loading, and managing AI models"""

    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        self.models_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.whisper_dir = self.models_dir / "whisper"
        self.llama_dir = self.models_dir / "llama"
        self.transformers_dir = self.models_dir / "transformers"
        self.persian_dir = self.models_dir / "persian"

        for directory in [
            self.whisper_dir,
            self.llama_dir,
            self.transformers_dir,
            self.persian_dir,
        ]:
            directory.mkdir(exist_ok=True)

        # Model configurations with detailed metadata
        self.model_configs = {
            "whisper": {
                "tiny": {
                    "size_mb": 39,
                    "accuracy": "~32% WER on LibriSpeech",
                    "speed": "~32x realtime",
                    "languages": 99,
                    "parameters": "39M",
                },
                "base": {
                    "size_mb": 74,
                    "accuracy": "~21% WER on LibriSpeech",
                    "speed": "~16x realtime",
                    "languages": 99,
                    "parameters": "74M",
                },
                "small": {
                    "size_mb": 244,
                    "accuracy": "~15% WER on LibriSpeech",
                    "speed": "~6x realtime",
                    "languages": 99,
                    "parameters": "244M",
                },
                "medium": {
                    "size_mb": 769,
                    "accuracy": "~12% WER on LibriSpeech",
                    "speed": "~2x realtime",
                    "languages": 99,
                    "parameters": "769M",
                },
                "large": {
                    "size_mb": 1550,
                    "accuracy": "~9% WER on LibriSpeech",
                    "speed": "~1x realtime",
                    "languages": 99,
                    "parameters": "1550M",
                },
            },
            "llama": {
                "llama-2-7b-chat": {
                    "size_mb": 4000,
                    "size_gb": 3.9,
                    "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
                    "filename": "llama-2-7b-chat.Q4_K_M.gguf",
                    "quantization": "Q4_0",
                    "parameters": "7B",
                    "context_length": 4096,
                    "description": "Llama 2 7B Chat model optimized for conversation",
                },
                "llama-2-13b-chat": {
                    "size_mb": 7000,
                    "size_gb": 6.8,
                    "repo_id": "TheBloke/Llama-2-13B-Chat-GGUF",
                    "filename": "llama-2-13b-chat.Q4_K_M.gguf",
                    "quantization": "Q4_0",
                    "parameters": "13B",
                    "context_length": 4096,
                    "description": "Llama 2 13B Chat model with better performance",
                },
            },
            "transformers": {
                "sentiment": {
                    "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "size_mb": 500,
                    "task": "sentiment-analysis",
                    "languages": ["en"],
                    "description": "RoBERTa model fine-tuned for sentiment analysis",
                },
                "emotion": {
                    "model_name": "j-hartmann/emotion-english-distilroberta-base",
                    "size_mb": 250,
                    "task": "emotion-classification",
                    "languages": ["en"],
                    "emotions": [
                        "anger",
                        "disgust",
                        "fear",
                        "joy",
                        "neutral",
                        "sadness",
                        "surprise",
                    ],
                    "description": "DistilRoBERTa model for emotion classification",
                },
                "multilingual": {
                    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "size_mb": 400,
                    "task": "sentence-embedding",
                    "languages": [
                        "en",
                        "de",
                        "fr",
                        "it",
                        "es",
                        "pl",
                        "tr",
                        "ru",
                        "bg",
                        "ro",
                        "ar",
                        "sw",
                        "th",
                    ],
                    "description": "Multilingual sentence embeddings",
                },
                "persian_sentiment": {
                    "model_name": "HooshvareLab/bert-fa-base-uncased-sentiment-persent",
                    "size_mb": 400,
                    "task": "sentiment-analysis",
                    "languages": ["fa"],
                    "description": "Persian BERT model for sentiment analysis",
                },
            },
        }

        # Download progress tracking
        self._download_progress = {}
        self._download_lock = threading.Lock()

        logger.info(f"ModelManager initialized. Models directory: {self.models_dir}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about available and installed models"""
        info = {
            "whisper": {
                "available_models": list(self.model_configs["whisper"].keys()),
                "current_model": getattr(settings.model, "whisper_model", "base"),
                "installed": self.is_whisper_available(),
                "installation_path": str(self.whisper_dir),
                "total_size_mb": sum(
                    config["size_mb"]
                    for config in self.model_configs["whisper"].values()
                ),
            },
            "llama": {
                "available_models": list(self.model_configs["llama"].keys()),
                "current_model": getattr(
                    settings.model, "llama_model", "llama-2-7b-chat"
                ),
                "installed": self.is_llama_available(),
                "installation_path": str(self.llama_dir),
                "total_size_mb": sum(
                    config["size_mb"] for config in self.model_configs["llama"].values()
                ),
            },
            "transformers": {
                "available_models": list(self.model_configs["transformers"].keys()),
                "installed": self.are_transformers_available(),
                "installation_path": str(self.transformers_dir),
                "total_size_mb": sum(
                    config["size_mb"]
                    for config in self.model_configs["transformers"].values()
                ),
            },
        }

        # Add detailed installation status
        for model_type in info:
            info[model_type]["detailed_status"] = self._get_detailed_status(model_type)

        return info

    def _get_detailed_status(self, model_type: str) -> Dict:
        """Get detailed status for a model type"""
        status = {"installed_models": [], "missing_models": [], "corrupted_models": []}

        if model_type == "whisper":
            for model_name in self.model_configs["whisper"]:
                if self.is_whisper_available(model_name):
                    status["installed_models"].append(model_name)
                else:
                    status["missing_models"].append(model_name)

        elif model_type == "llama":
            for model_name in self.model_configs["llama"]:
                config = self.model_configs["llama"][model_name]
                model_path = self.llama_dir / config["filename"]
                if (
                    model_path.exists() and model_path.stat().st_size > 1024 * 1024
                ):  # > 1MB
                    status["installed_models"].append(model_name)
                else:
                    status["missing_models"].append(model_name)

        elif model_type == "transformers":
            for model_name, config in self.model_configs["transformers"].items():
                model_dir = self.transformers_dir / model_name
                if model_dir.exists() and (model_dir / "config.json").exists():
                    status["installed_models"].append(model_name)
                else:
                    status["missing_models"].append(model_name)

        return status

    def is_whisper_available(self, model_name: str = None) -> bool:
        """Check if Whisper model is available"""
        if not HAS_WHISPER:
            return False

        model_name = model_name or getattr(settings.model, "whisper_model", "base")

        try:
            # Check if model exists in whisper cache or our directory
            import whisper

            # Try to load the model (this will download if needed)
            model_path = whisper._download(whisper._MODELS[model_name], root=str(self.whisper_dir), in_memory=False)
            return Path(model_path).exists()
        except Exception as e:
            logger.debug(f"Whisper model check failed: {e}")
            return False

    def download_whisper_model(self, model_name: str = None) -> bool:
        """Download Whisper model with progress tracking"""
        if not HAS_WHISPER:
            logger.error("Whisper library not available")
            return False

        model_name = model_name or getattr(settings.model, 'whisper_model', 'base')

        if model_name not in self.model_configs["whisper"]:
            logger.error(f"Unknown Whisper model: {model_name}")
            return False

        try:
            logger.info(f"Downloading Whisper model: {model_name}")

            config = self.model_configs["whisper"][model_name]
            logger.info(f"Model info: {config['size_mb']}MB, {config['accuracy']}")

            import whisper

            with self._download_lock:
                self._download_progress[f"whisper_{model_name}"] = {"status": "downloading", "progress": 0}

            # Remove download_root argument!
            model_path = whisper._download(
                whisper._MODELS[model_name],
                root=str(self.whisper_dir),
                in_memory=False
            )

            if Path(model_path).exists():
                file_size = Path(model_path).stat().st_size
                logger.info(f"Whisper model downloaded: {model_path} ({file_size:,} bytes)")

                with self._download_lock:
                    self._download_progress[f"whisper_{model_name}"] = {"status": "completed", "progress": 100}

                return True
            else:
                logger.error(f"Download failed: model file not found")
                with self._download_lock:
                    self._download_progress[f"whisper_{model_name}"] = {"status": "failed", "progress": 0}
                return False

        except Exception as e:
            logger.error(f"Failed to download Whisper model {model_name}: {e}")
            with self._download_lock:
                self._download_progress[f"whisper_{model_name}"] = {"status": "failed", "error": str(e)}
            return False

    def is_llama_available(self, model_name: str = None) -> bool:
        """Check if LLaMA model is available"""
        model_name = model_name or getattr(
            settings.model, "llama_model", "llama-2-7b-chat"
        )

        if model_name not in self.model_configs["llama"]:
            return False

        config = self.model_configs["llama"][model_name]
        model_path = self.llama_dir / config["filename"]

        # Check if file exists and has reasonable size (> 100MB)
        if model_path.exists():
            file_size = model_path.stat().st_size
            min_size = 100 * 1024 * 1024  # 100MB minimum
            return file_size > min_size

        return False

    def download_llama_model(self, model_name: str = None) -> bool:
        """Download LLaMA model from Hugging Face"""
        if not HAS_HF_HUB:
            logger.error("Hugging Face Hub not available")
            return False

        model_name = model_name or getattr(
            settings.model, "llama_model", "llama-2-7b-chat"
        )

        if model_name not in self.model_configs["llama"]:
            logger.error(f"Unknown LLaMA model: {model_name}")
            return False

        config = self.model_configs["llama"][model_name]

        try:
            logger.info(f"Downloading LLaMA model: {model_name}")
            logger.info(f"Repository: {config['repo_id']}")
            logger.info(f"File: {config['filename']} (~{config['size_gb']}GB)")

            # Check disk space
            required_space = config["size_mb"] * 1024 * 1024 * 1.2  # 20% buffer
            free_space = shutil.disk_usage(self.llama_dir).free

            if free_space < required_space:
                logger.error(
                    f"Insufficient disk space. Required: {required_space/(1024**3):.1f}GB"
                )
                return False

            with self._download_lock:
                self._download_progress[f"llama_{model_name}"] = {
                    "status": "downloading",
                    "progress": 0,
                }

            # Download from Hugging Face
            model_path = hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename"],
                local_dir=str(self.llama_dir),
                resume_download=True,
                cache_dir=str(self.llama_dir / ".cache"),
            )

            # Verify download
            if Path(model_path).exists():
                file_size = Path(model_path).stat().st_size
                logger.info(
                    f"LLaMA model downloaded: {model_path} ({file_size/(1024**3):.2f}GB)"
                )

                with self._download_lock:
                    self._download_progress[f"llama_{model_name}"] = {
                        "status": "completed",
                        "progress": 100,
                    }

                return True
            else:
                logger.error("LLaMA download failed: file not found")
                with self._download_lock:
                    self._download_progress[f"llama_{model_name}"] = {
                        "status": "failed",
                        "progress": 0,
                    }
                return False

        except Exception as e:
            logger.error(f"Failed to download LLaMA model: {e}")
            with self._download_lock:
                self._download_progress[f"llama_{model_name}"] = {
                    "status": "failed",
                    "error": str(e),
                }
            return False

    def are_transformers_available(self) -> bool:
        """Check if transformer models are available"""
        if not HAS_TRANSFORMERS:
            return False

        # Check if at least one transformer model is installed
        for model_name in self.model_configs["transformers"]:
            model_dir = self.transformers_dir / model_name
            if model_dir.exists() and (model_dir / "config.json").exists():
                return True

        return False

    def download_transformer_model(self, model_type: str) -> bool:
        """Download a specific transformer model"""
        if not HAS_TRANSFORMERS:
            logger.error("Transformers library not available")
            return False

        if model_type not in self.model_configs["transformers"]:
            logger.error(f"Unknown transformer model type: {model_type}")
            return False

        config = self.model_configs["transformers"][model_type]
        model_name = config["model_name"]

        try:
            logger.info(f"Downloading transformer model: {model_type}")
            logger.info(f"Model: {model_name} (~{config['size_mb']}MB)")

            model_dir = self.transformers_dir / model_type
            model_dir.mkdir(exist_ok=True)

            with self._download_lock:
                self._download_progress[f"transformer_{model_type}"] = {
                    "status": "downloading",
                    "progress": 0,
                }

            # Download tokenizer and model
            from transformers import AutoTokenizer, AutoModel

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

            # Save to local directory
            tokenizer.save_pretrained(str(model_dir))
            model.save_pretrained(str(model_dir))

            # Save metadata
            metadata = {
                "model_name": model_name,
                "model_type": model_type,
                "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": config,
            }

            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Transformer model saved to: {model_dir}")

            with self._download_lock:
                self._download_progress[f"transformer_{model_type}"] = {
                    "status": "completed",
                    "progress": 100,
                }

            return True

        except Exception as e:
            logger.error(f"Failed to download transformer model {model_type}: {e}")
            with self._download_lock:
                self._download_progress[f"transformer_{model_type}"] = {
                    "status": "failed",
                    "error": str(e),
                }
            return False

    def download_alternative_models(self) -> bool:
        """Download smaller alternative models for limited resources"""
        try:
            logger.info("Downloading alternative lightweight models...")

            success_count = 0
            total_models = len(self.model_configs["transformers"])

            # Download basic transformer models
            for model_type in ["sentiment", "emotion"]:
                if self.download_transformer_model(model_type):
                    success_count += 1

            # Download sentence transformers if available
            try:
                from sentence_transformers import SentenceTransformer

                models_to_download = [
                    ("all-MiniLM-L6-v2", "Lightweight sentence embeddings"),
                    ("paraphrase-multilingual-MiniLM-L12-v2", "Multilingual support"),
                ]

                for model_name, description in models_to_download:
                    try:
                        model_path = (
                            self.transformers_dir
                            / "sentence_transformers"
                            / model_name.replace("/", "_")
                        )

                        if not model_path.exists():
                            logger.info(f"Downloading {description}: {model_name}")
                            model = SentenceTransformer(model_name)
                            model.save(str(model_path))
                            logger.info(f"Downloaded: {model_name}")
                            success_count += 1
                        else:
                            logger.info(f"Already exists: {model_name}")
                            success_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to download {model_name}: {e}")

            except ImportError:
                logger.warning("sentence-transformers not available")

            logger.info(
                f"Alternative models download completed: {success_count} models"
            )
            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to download alternative models: {e}")
            return False

    def install_for_iran(self) -> bool:
        """Install models optimized for Iran (CPU-only, smaller models)"""
        logger.info("Installing models optimized for Iran/CPU-only systems...")

        success_components = []

        try:
            # Use smaller Whisper models for better CPU performance
            iran_whisper_models = ["base", "small"]

            for model in iran_whisper_models:
                if self.download_whisper_model(model):
                    success_components.append(f"whisper-{model}")
                    logger.info(f"✅ Whisper {model} model installed")
                else:
                    logger.warning(f"⚠️ Failed to install Whisper {model}")

            # Download basic transformer models (excluding large ones)
            basic_transformers = ["sentiment", "emotion"]

            for model_type in basic_transformers:
                if self.download_transformer_model(model_type):
                    success_components.append(f"transformer-{model_type}")
                    logger.info(f"✅ {model_type} model installed")

            # Download alternative lightweight models
            if self.download_alternative_models():
                success_components.append("alternative-models")
                logger.info("✅ Alternative models installed")

            # Update settings for CPU optimization
            if hasattr(settings.model, "whisper_model"):
                settings.model.whisper_model = "small"
            if hasattr(settings.model, "device"):
                settings.model.device = "cpu"
            if hasattr(settings.model, "quantization"):
                settings.model.quantization = True

            logger.info(
                f"Iran-optimized installation complete: {len(success_components)} components installed"
            )
            logger.info(f"Installed: {', '.join(success_components)}")

            return len(success_components) > 0

        except Exception as e:
            logger.error(f"Iran installation failed: {e}")
            return False

    def estimate_disk_space(self, models: List[str] = None) -> Dict[str, Any]:
        """Estimate required disk space for models"""
        if models is None:
            # Use default models based on current settings
            models = [
                getattr(settings.model, "whisper_model", "base"),
                getattr(settings.model, "llama_model", "llama-2-7b-chat"),
            ]

        total_size_mb = 0
        size_breakdown = {}

        for model in models:
            # Check Whisper models
            if model in self.model_configs["whisper"]:
                size_mb = self.model_configs["whisper"][model]["size_mb"]
                size_breakdown[f"whisper-{model}"] = f"{size_mb} MB"
                total_size_mb += size_mb

            # Check LLaMA models
            elif model in self.model_configs["llama"]:
                size_mb = self.model_configs["llama"][model]["size_mb"]
                size_breakdown[f"llama-{model}"] = f"{size_mb} MB"
                total_size_mb += size_mb
            # Check Transformers models
            elif model in self.model_configs["transformers"]:
                size_mb = self.model_configs["transformers"][model]["size_mb"]
                size_breakdown[f"transformer-{model}"] = f"{size_mb} MB"
                total_size_mb += size_mb
            else:
                logger.warning(f"Unknown model: {model}")
        return {"total_size_mb": total_size_mb, "size_breakdown": size_breakdown}

    def get_download_progress(self) -> Dict[str, Any]:
        """Get current download progress for all models"""
        with self._download_lock:
            return self._download_progress.copy()

    def clear_download_progress(self) -> None:
        """Clear the download progress tracking"""
        with self._download_lock:
            self._download_progress.clear()
            logger.info("Download progress cleared")

    def reset_model_configs(self) -> None:
        """Reset model configurations to default"""
        self.model_configs = {
            "whisper": {
                "tiny": {
                    "size_mb": 39,
                    "accuracy": "~32% WER on LibriSpeech",
                    "speed": "~32x realtime",
                    "languages": 99,
                    "parameters": "39M",
                },
                "base": {
                    "size_mb": 74,
                    "accuracy": "~21% WER on LibriSpeech",
                    "speed": "~16x realtime",
                    "languages": 99,
                    "parameters": "74M",
                },
                "small": {
                    "size_mb": 244,
                    "accuracy": "~15% WER on LibriSpeech",
                    "speed": "~6x realtime",
                    "languages": 99,
                    "parameters": "244M",
                },
                "medium": {
                    "size_mb": 769,
                    "accuracy": "~12% WER on LibriSpeech",
                    "speed": "~2x realtime",
                    "languages": 99,
                    "parameters": "769M",
                },
                "large": {
                    "size_mb": 1550,
                    "accuracy": "~9% WER on LibriSpeech",
                    "speed": "~1x realtime",
                    "languages": 99,
                    "parameters": "1550M",
                },
            },
            # Other model configurations...
        }
        logger.info("Model configurations reset to default")

    def get_model_config(
        self, model_type: str, model_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model"""
        if model_type not in self.model_configs:
            logger.error(f"Unknown model type: {model_type}")
            return None

        if model_name not in self.model_configs[model_type]:
            logger.error(f"Unknown model name: {model_name} for type {model_type}")
            return None

        return self.model_configs[model_type][model_name]

    def get_model_directory(self, model_type: str, model_name: str) -> Optional[Path]:
        """Get the directory path for a specific model"""
        if model_type not in self.model_configs:
            logger.error(f"Unknown model type: {model_type}")
            return None

        if model_name not in self.model_configs[model_type]:
            logger.error(f"Unknown model name: {model_name} for type {model_type}")
            return None

        if model_type == "whisper":
            return self.whisper_dir
        elif model_type == "llama":
            return self.llama_dir
        elif model_type == "transformers":
            return self.transformers_dir
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None

    def get_model_file_path(self, model_type: str, model_name: str) -> Optional[Path]:
        """Get the file path for a specific model"""
        model_dir = self.get_model_directory(model_type, model_name)
        if not model_dir:
            return None

        config = self.get_model_config(model_type, model_name)
        if not config:
            return None

        if model_type == "whisper":
            return model_dir / f"{model_name}.pt"
        elif model_type == "llama":
            return model_dir / config["filename"]
        elif model_type == "transformers":
            return model_dir / "pytorch_model.bin"
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None

    def delete_model(self, model_type: str, model_name: str) -> bool:
        """Delete a specific model and its files"""
        model_dir = self.get_model_directory(model_type, model_name)
        if not model_dir:
            logger.error(f"Model directory not found for {model_type}/{model_name}")
            return False

        try:
            if model_type == "whisper":
                model_file = model_dir / f"{model_name}.pt"
            elif model_type == "llama":
                config = self.get_model_config(model_type, model_name)
                if not config:
                    logger.error(
                        f"Model config not found for {model_type}/{model_name}"
                    )
                    return False
                model_file = model_dir / config["filename"]
            elif model_type == "transformers":
                model_file = model_dir / "pytorch_model.bin"
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return False

            if model_file.exists():
                model_file.unlink()
                logger.info(f"Deleted {model_file}")

            # Remove empty directory
            if not any(model_dir.iterdir()):
                model_dir.rmdir()
                logger.info(f"Removed empty directory: {model_dir}")

            return True
        except Exception as e:
            logger.error(f"Failed to delete {model_type}/{model_name}: {e}")
            return False

    def clear_all_models(self) -> bool:
        """Clear all downloaded models and their directories"""
        try:
            logger.info("Clearing all models...")
            shutil.rmtree(self.models_dir)
            self.models_dir.mkdir(exist_ok=True)  # Recreate the main directory
            logger.info("All models cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear all models: {e}")
            return False

    def get_model_hash(self, model_type: str, model_name: str) -> Optional[str]:
        """Get the SHA256 hash of a model file"""
        model_file = self.get_model_file_path(model_type, model_name)
        if not model_file or not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            return None

        try:
            sha256_hash = hashlib.sha256()
            with open(model_file, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute hash for {model_type}/{model_name}: {e}")
            return None

    get_installed_models = get_model_info
    """Alias for backward compatibility"""


# import logging
# import threading
# import shutil
# import time
# import json
# from pathlib import Path
# from typing import Dict, Any, List, Optional
# from huggingface_hub import hf_hub_download
# from config.settings import settings
# # Initialize logging
# logger = logging.getLogger(__name__)
# # Set up logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('model_manager.log', mode='a')
#     ]
# )
# # Check for required libraries
# try:
#     import whisper
#     HAS_WHISPER = True
# except ImportError:
#     HAS_WHISPER = False
#     logger.warning("Whisper library not found. Install it with: pip install git+https://github.com/openai/whisper.git")
# try:
#     from huggingface_hub import hf_hub_download
#     HAS_HF_HUB = True
# except ImportError:
#     HAS_HF_HUB = False
#     logger.warning("Hugging Face Hub library not found. Install it with: pip install huggingface_hub")
# try:
#     from transformers import AutoTokenizer, AutoModel
