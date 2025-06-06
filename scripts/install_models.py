#!/usr/bin/env python3
"""
Model Installation Script for Phone Call Analysis System
Downloads and sets up required AI models with comprehensive error handling and progress tracking
"""

import sys
import os
import time
import requests
import json
from pathlib import Path
import argparse
import shutil
import hashlib
import zipfile
import tarfile
from urllib.parse import urlparse
from typing import Dict, List, Optional, Tuple
import threading
import queue

# Try to import optional dependencies
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

try:
    import huggingface_hub
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.model_manager import ModelManager
    from loguru import logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

# Model URLs and configurations
MODEL_CONFIGS = {
    "whisper": {
        "tiny": {
            "size_mb": 39,
            "url": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22794/tiny.pt",
            "hash": "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22794"
        },
        "base": {
            "size_mb": 74,
            "url": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
            "hash": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e"
        },
        "small": {
            "size_mb": 244,
            "url": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
            "hash": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"
        },
        "medium": {
            "size_mb": 769,
            "url": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
            "hash": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1"
        },
        "large": {
            "size_mb": 1550,
            "url": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",
            "hash": "e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a"
        }
    },
    "llama": {
        "llama-2-7b-chat": {
            "size_mb": 4000,
            "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
            "filename": "llama-2-7b-chat.Q4_K_M.gguf",
            "quantized": True
        }
    },
    "transformers": {
        "sentiment": {
            "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "size_mb": 500
        },
        "emotion": {
            "model_name": "j-hartmann/emotion-english-distilroberta-base",
            "size_mb": 250
        },
        "multilingual": {
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "size_mb": 400
        }
    }
}

def setup_logging(verbose: bool = False):
    """Setup logging for installation"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Also log to file
    log_file = project_root / "logs" / "installation.log"
    log_file.parent.mkdir(exist_ok=True)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="10 MB"
    )

def check_internet_connection(timeout: int = 10) -> bool:
    """Check if internet connection is available"""
    test_urls = [
        "https://httpbin.org/get",
        "https://www.google.com",
        "https://huggingface.co",
        "https://github.com"
    ]
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                logger.info(f"✅ Internet connection verified via {url}")
                return True
        except:
            continue
    
    logger.error("❌ No internet connection detected")
    return False

def check_disk_space(required_gb: float = 15.0) -> Tuple[bool, float]:
    """Check available disk space"""
    try:
        total, used, free = shutil.disk_usage(project_root)
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        
        logger.info(f"Disk usage: {used_gb:.1f}GB used / {total_gb:.1f}GB total ({free_gb:.1f}GB free)")
        
        if free_gb < required_gb:
            logger.warning(f"⚠️ Low disk space! Required: {required_gb}GB, Available: {free_gb:.1f}GB")
            return False, free_gb
        
        logger.info(f"✅ Sufficient disk space available: {free_gb:.1f}GB")
        return True, free_gb
        
    except Exception as e:
        logger.error(f"Failed to check disk space: {e}")
        return True, 0.0  # Assume OK if we can't check

def download_file_with_progress(url: str, destination: Path, description: str = None, 
                               expected_hash: str = None, chunk_size: int = 8192) -> bool:
    """Download file with progress bar and verification"""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists and is valid
        if destination.exists() and expected_hash:
            if verify_file_hash(destination, expected_hash):
                logger.info(f"✅ File already exists and verified: {destination.name}")
                return True
            else:
                logger.info(f"🔄 File exists but hash mismatch, re-downloading: {destination.name}")
                destination.unlink()
        
        logger.info(f"📥 Downloading: {url}")
        
        # Start download
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        desc = description or f"Downloading {destination.name}"
        
        # Use progress bar if available
        if HAS_TQDM and total_size > 0:
            with open(destination, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress_bar.update(len(chunk))
        else:
            # Fallback without progress bar
            with open(destination, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r{desc}: {percent:.1f}%", end='', flush=True)
                print()  # New line after download
        
        # Verify download
        if not destination.exists():
            logger.error(f"❌ Download failed: file not created")
            return False
        
        file_size = destination.stat().st_size
        if total_size > 0 and file_size != total_size:
            logger.error(f"❌ Download incomplete: {file_size} vs {total_size} bytes")
            destination.unlink()
            return False
        
        # Verify hash if provided
        if expected_hash and not verify_file_hash(destination, expected_hash):
            logger.error(f"❌ Hash verification failed for {destination.name}")
            destination.unlink()
            return False
        
        logger.info(f"✅ Downloaded successfully: {destination.name} ({file_size:,} bytes)")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Network error downloading {url}: {e}")
        if destination.exists():
            destination.unlink()
        return False
    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        if destination.exists():
            destination.unlink()
        return False

def verify_file_hash(file_path: Path, expected_hash: str, algorithm: str = "sha1") -> bool:
    """Verify file hash"""
    try:
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        actual_hash = hash_func.hexdigest()
        
        if actual_hash == expected_hash:
            logger.debug(f"✅ Hash verified: {file_path.name}")
            return True
        else:
            logger.error(f"❌ Hash mismatch for {file_path.name}")
            logger.error(f"   Expected: {expected_hash}")
            logger.error(f"   Actual:   {actual_hash}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Hash verification failed: {e}")
        return False

def download_huggingface_model(repo_id: str, filename: str, local_dir: Path) -> bool:
    """Download model from Hugging Face Hub"""
    try:
        if not HAS_HF_HUB:
            logger.error("❌ huggingface_hub not available. Install with: pip install huggingface_hub")
            return False
        
        from huggingface_hub import hf_hub_download
        
        logger.info(f"📥 Downloading from Hugging Face: {repo_id}/{filename}")
        
        local_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            resume_download=True
        )
        
        logger.info(f"✅ Downloaded: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download from Hugging Face: {e}")
        return False

def install_whisper_models(model_manager: ModelManager, models: List[str]) -> bool:
    """Install Whisper speech recognition models"""
    logger.info(f"🎤 Installing Whisper models: {', '.join(models)}")
    
    success_count = 0
    total_size_mb = 0
    
    for model in models:
        config = MODEL_CONFIGS["whisper"].get(model)
        if not config:
            logger.error(f"❌ Unknown Whisper model: {model}")
            continue
        
        size_mb = config["size_mb"]
        total_size_mb += size_mb
        
        logger.info(f"📦 Installing Whisper '{model}' model ({size_mb}MB)...")
        
        try:
            if model_manager.download_whisper_model(model):
                logger.info(f"✅ Whisper '{model}' installed successfully")
                success_count += 1
            else:
                logger.error(f"❌ Failed to install Whisper '{model}' model")
        except Exception as e:
            logger.error(f"❌ Error installing Whisper '{model}': {e}")
    
    logger.info(f"📊 Whisper installation summary: {success_count}/{len(models)} models installed ({total_size_mb}MB total)")
    
    return success_count > 0

def install_llama_models(model_manager: ModelManager) -> bool:
    """Install LLaMA language models"""
    logger.info("🧠 Installing LLaMA models...")
    
    # Check disk space requirement
    required_gb = 5.0
    has_space, free_gb = check_disk_space(required_gb)
    
    if not has_space:
        logger.warning(f"⚠️ Insufficient disk space for LLaMA ({required_gb}GB required, {free_gb:.1f}GB available)")
        logger.info("Skipping LLaMA installation. The system will work with limited features.")
        return False
    
    try:
        if model_manager.download_llama_model():
            logger.info("✅ LLaMA model installed successfully")
            return True
        else:
            logger.warning("❌ Failed to install LLaMA model")
            logger.info("LLaMA model is optional. The system will work with basic features.")
            return False
    except Exception as e:
        logger.error(f"❌ LLaMA installation error: {e}")
        return False

def install_transformer_models(model_manager: ModelManager) -> bool:
    """Install transformer-based models for sentiment and emotion analysis"""
    logger.info("🤖 Installing transformer models...")
    
    success_count = 0
    models = MODEL_CONFIGS["transformers"]
    
    for model_type, config in models.items():
        model_name = config["model_name"]
        size_mb = config["size_mb"]
        
        logger.info(f"📦 Installing {model_type} model: {model_name} ({size_mb}MB)")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Save to local cache
            cache_dir = project_root / "models" / "transformers" / model_type
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            tokenizer.save_pretrained(str(cache_dir))
            model.save_pretrained(str(cache_dir))
            
            logger.info(f"✅ {model_type} model installed successfully")
            success_count += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to install {model_type} model: {e}")
    
    logger.info(f"📊 Transformer models: {success_count}/{len(models)} installed")
    return success_count > 0

def install_persian_support() -> bool:
    """Install Persian language support files"""
    logger.info("🇮🇷 Installing Persian language support...")
    
    try:
        persian_dir = project_root / "models" / "persian"
        persian_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Persian profanity dictionary
        persian_profanity = [
            # Basic Persian profanity words (example list)
            "کسکش", "جنده", "گوه", "کیری", "کونی", "آشغال",
            "احمق", "خر", "گاو", "سگ", "الاغ"
        ]
        
        profanity_file = persian_dir / "profanity.txt"
        with open(profanity_file, 'w', encoding='utf-8') as f:
            for word in persian_profanity:
                f.write(f"{word}\n")
        
        # Create Persian stopwords
        persian_stopwords = [
            "در", "از", "به", "با", "که", "این", "آن", "را", "و", "یا",
            "تا", "بر", "برای", "است", "بود", "شد", "می", "خواهد"
        ]
        
        stopwords_file = persian_dir / "stopwords.txt"
        with open(stopwords_file, 'w', encoding='utf-8') as f:
            for word in persian_stopwords:
                f.write(f"{word}\n")
        
        # Create configuration file
        config = {
            "language": "persian",
            "encoding": "utf-8",
            "profanity_file": str(profanity_file),
            "stopwords_file": str(stopwords_file),
            "supported_dialects": ["standard", "tehrani", "isfahani"]
        }
        
        config_file = persian_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Persian language support installed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to install Persian support: {e}")
        return False

def verify_installation(model_manager: ModelManager) -> bool:
    """Verify that models are properly installed and working"""
    logger.info("🔍 Verifying installation...")
    
    verification_results = {
        "whisper": False,
        "transformers": False,
        "llama": False,
        "python_imports": False
    }
    
    # Test Python imports
    try:
        import torch
        import whisper
        import transformers
        import pyaudio
        import librosa
        
        logger.info("✅ Core Python packages imported successfully")
        verification_results["python_imports"] = True
    except ImportError as e:
        logger.error(f"❌ Failed to import core packages: {e}")
    
    # Test Whisper
    try:
        if model_manager.is_whisper_available():
            # Try loading a model
            import whisper
            test_model = whisper.load_model("base")
            if test_model:
                logger.info("✅ Whisper model loads successfully")
                verification_results["whisper"] = True
        else:
            logger.error("❌ Whisper model not found")
    except Exception as e:
        logger.error(f"❌ Whisper verification failed: {e}")
    
    # Test LLaMA (optional)
    try:
        if model_manager.is_llama_available():
            logger.info("✅ LLaMA model found")
            verification_results["llama"] = True
        else:
            logger.warning("⚠️ LLaMA model not found (optional)")
    except Exception as e:
        logger.warning(f"⚠️ LLaMA verification failed: {e}")
    
    # Test transformers
    try:
        from transformers import pipeline
        test_classifier = pipeline("sentiment-analysis", 
                                  model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        if test_classifier:
            logger.info("✅ Transformer models load successfully")
            verification_results["transformers"] = True
    except Exception as e:
        logger.error(f"❌ Transformer verification failed: {e}")
    
    # Overall verification
    critical_components = ["whisper", "python_imports"]
    critical_passed = all(verification_results[comp] for comp in critical_components)
    
    if critical_passed:
        logger.info("✅ Critical components verified successfully")
        return True
    else:
        logger.error("❌ Critical component verification failed")
        return False

def perform_system_checks() -> Dict[str, bool]:
    """Perform comprehensive system checks"""
    logger.info("🔧 Performing system checks...")
    
    checks = {
        "python_version": False,
        "disk_space": False,
        "internet": False,
        "dependencies": False,
        "permissions": False
    }
    
    # Python version check
    if sys.version_info >= (3, 8):
        logger.info(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        checks["python_version"] = True
    else:
        logger.error(f"❌ Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Disk space check
    checks["disk_space"], _ = check_disk_space(15.0)
    
    # Internet check
    checks["internet"] = check_internet_connection()
    
    # Dependencies check
    required_packages = [
        "torch", "transformers", "librosa", "pyaudio", 
        "loguru", "requests", "numpy", "scipy"
    ]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.debug(f"✅ {package}")
        except ImportError:
            logger.warning(f"⚠️ {package} not found")
            missing_packages.append(package)
    
    if not missing_packages:
        checks["dependencies"] = True
        logger.info("✅ All required packages available")
    else:
        logger.warning(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        logger.info("Install with: pip install -r requirements.txt")
    
    # Permissions check
    try:
        test_file = project_root / "test_permissions.tmp"
        test_file.write_text("test")
        test_file.unlink()
        checks["permissions"] = True
        logger.info("✅ Write permissions verified")
    except Exception as e:
        logger.error(f"❌ Permission error: {e}")
    
    # Summary
    passed_checks = sum(checks.values())
    total_checks = len(checks)
    logger.info(f"📊 System checks: {passed_checks}/{total_checks} passed")
    
    return checks

def create_installation_report(model_manager, install_start_time, success, installed_components):
    """Create detailed installation report"""
    install_duration = time.time() - install_start_time
    
    # Get model information
    try:
        model_info = model_manager.get_model_info()
        disk_info = model_manager.estimate_disk_space()
        total_size_mb = disk_info.get("total_size_mb", 0)
        total_size_gb = round(total_size_mb / 1024, 2)
    except:
        model_info = {}
        disk_usage = {"total_size_gb": 0, "breakdown": {}}
    
    # System information
    import platform
    system_info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_executable": sys.executable
    }
    
    # Create report
    report = {
        "installation": {
            "success": success,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(install_duration, 2),
            "duration_minutes": round(install_duration / 60, 2),
            "installed_components": installed_components
        },
        "models": model_info,
        "disk_usage": {
            "total_size_mb": total_size_mb,
            "total_size_gb": total_size_gb,
            "size_breakdown": disk_info.get("size_breakdown", {}),
        },
        "system": system_info,
        "versions": {
            "installer_version": "1.0.0",
            "project_version": "1.0.0"
        }
    }
    
    # Save report
    try:
        report_file = project_root / "installation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 Installation report saved: {report_file}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save installation report: {e}")
    
    return report

def show_installation_summary(report: Dict, success: bool):
    """Show installation summary and next steps"""
    logger.info("\n" + "=" * 60)
    
    if success:
        logger.info("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        # Installation details
        duration = report["installation"]["duration_minutes"]
        components = report["installation"]["installed_components"]
        disk_gb = report["disk_usage"]["total_size_gb"]
        
        logger.info(f"⏱️  Installation time: {duration} minutes")
        logger.info(f"💾 Disk usage: {disk_gb:.1f} GB")
        logger.info(f"📦 Components installed: {', '.join(components)}")
        
        # Next steps
        logger.info("\n🚀 NEXT STEPS:")
        logger.info("1. Run GUI: python src/main.py")
        logger.info("2. Run CLI: python src/main.py --cli audio.wav")
        logger.info("3. Check status: python src/main.py --check-deps")
        logger.info("4. Read docs: docs/documentation.pdf")
        
        # Performance tips
        logger.info("\n💡 PERFORMANCE TIPS:")
        logger.info("• Close other applications for better performance")
        logger.info("• Use smaller models (base/small) for real-time analysis")
        logger.info("• Monitor system resources during operation")
        logger.info("• Enable model quantization in settings for lower memory usage")
        
    else:
        logger.error("❌ INSTALLATION FAILED")
        logger.error("=" * 60)
        
        logger.info("\n🔧 TROUBLESHOOTING:")
        logger.info("1. Check internet connection")
        logger.info("2. Ensure 15+ GB free disk space")
        logger.info("3. Verify Python 3.8+ installation")
        logger.info("4. Install dependencies: pip install -r requirements.txt")
        logger.info("5. Run system setup: python scripts/setup_environment.py")
        logger.info("6. Check logs in logs/installation.log")
        logger.info("7. Try installing components individually")
        
    logger.info("=" * 60)

def main():
    """Main installation function"""
    parser = argparse.ArgumentParser(
        description="Install AI models for Phone Call Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Install all models
  %(prog)s --iran                   # Iran-optimized installation
  %(prog)s --whisper-only --models base small
  %(prog)s --check-only             # Check installation status
  %(prog)s --force --verbose        # Force reinstall with debug output
        """
    )
    
    parser.add_argument("--iran", action="store_true", 
                       help="Install optimized models for Iran (CPU-only, smaller models)")
    parser.add_argument("--whisper-only", action="store_true", 
                       help="Install only Whisper models")
    parser.add_argument("--llama-only", action="store_true", 
                       help="Install only LLaMA models")
    parser.add_argument("--transformers-only", action="store_true",
                       help="Install only transformer models")
    parser.add_argument("--models", nargs="+", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Specific Whisper models to install")
    parser.add_argument("--force", action="store_true", 
                       help="Force reinstallation of existing models")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check current installation status")
    parser.add_argument("--skip-verification", action="store_true", 
                       help="Skip installation verification")
    parser.add_argument("--no-progress", action="store_true", 
                       help="Disable progress bars")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose logging with debug information")
    parser.add_argument("--offline", action="store_true",
                       help="Skip internet-dependent installations")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    install_start_time = time.time()
    
    logger.info("🤖 Phone Call Analysis System - Model Installer v1.0")
    logger.info("=" * 60)
    
    # Perform system checks
    if not args.check_only:
        system_checks = perform_system_checks()
        critical_failed = not all([
            system_checks["python_version"],
            system_checks["permissions"]
        ])
        
        if critical_failed and not args.force:
            logger.error("❌ Critical system checks failed")
            logger.info("Use --force to continue anyway")
            sys.exit(1)
        elif critical_failed:
            logger.warning("⚠️ Continuing with failed system checks due to --force flag")
    
    # Initialize model manager
    try:
        model_manager = ModelManager()
        logger.info("✅ ModelManager initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ModelManager: {e}")
        sys.exit(1)
    # Check internet connection if not offline mode
    if not args.offline and not check_internet_connection():
        logger.error("❌ No internet connection. Cannot proceed with installation.")
        sys.exit(1)
    # Check disk space
    has_space, free_gb = check_disk_space(15.0)
    if not has_space:
        logger.error(f"❌ Insufficient disk space: {free_gb:.1f}GB available, 15GB required")
        sys.exit(1)
    if has_space:
        logger.info(f"✅ Sufficient disk space available: {free_gb:.1f}GB")
    existing_models = model_manager.get_installed_models()
    if existing_models:
        logger.info(f"🔍 Found existing models: {', '.join(existing_models)}")
        if not args.force:
            logger.info("Use --force to reinstall existing models")
        else:
            logger.info("🔄 Reinstalling existing models due to --force flag")
    # Check if models are already installed
    if args.check_only:
        logger.info("🔍 Checking current installation status...")
        if verify_installation(model_manager):
            logger.info("✅ All models are properly installed and verified")
        else:
            logger.error("❌ Some models failed verification")
        sys.exit(0)
    # Install models based on arguments
    installed_components = []
    success = True
    if args.whisper_only or not (args.llama_only or args.transformers_only):
        whisper_models = args.models if args.models else ["tiny", "base", "small", "medium", "large"]
        if install_whisper_models(model_manager, whisper_models):
            installed_components.append("Whisper models")
        else:
            success = False
    if args.llama_only or not (args.whisper_only or args.transformers_only):
        if install_llama_models(model_manager):
            installed_components.append("LLaMA model")
        else:
            success = False
    if args.transformers_only or not (args.whisper_only or args.llama_only):
        if install_transformer_models(model_manager):
            installed_components.append("Transformer models")
        else:
            success = False
    if args.iran:
        if install_persian_support():
            installed_components.append("Persian language support")
        else:
            success = False
    # Verify installation if not skipped
    if not args.skip_verification:
        if not verify_installation(model_manager):
            logger.error("❌ Installation verification failed")
            success = False
        else:
            logger.info("✅ Installation verified successfully")
    # Create installation report
    report = create_installation_report(
        model_manager, install_start_time, success, installed_components
    )
    # Show installation summary
    show_installation_summary(report, success)
    # Exit with appropriate status
    if success:
        logger.info("🎉 Installation completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Installation failed. Check logs for details.")
        sys.exit(1)
if __name__ == "__main__":
    main()