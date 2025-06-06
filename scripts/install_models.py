import os
import subprocess

def install_model(model_name):
    try:
        print(f"Installing {model_name}...")
        # Here you would typically have a command to install the model
        # For example, if using pip to install a package:
        subprocess.check_call([sys.executable, "-m", "pip", "install", model_name])
        print(f"{model_name} installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {model_name}. Error: {e}")

def main():
    models_to_install = [
        "whisper",  # Replace with actual model package name
        "llama"     # Replace with actual model package name
    ]
    
    for model in models_to_install:
        install_model(model)

if __name__ == "__main__":
    main()