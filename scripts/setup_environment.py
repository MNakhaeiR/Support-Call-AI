import os
import subprocess

def setup_virtual_environment():
    """Create a virtual environment for the project."""
    if not os.path.exists('venv'):
        subprocess.run(['python', '-m', 'venv', 'venv'])
        print("Virtual environment created.")
    else:
        print("Virtual environment already exists.")

def install_requirements():
    """Install the required packages from requirements.txt."""
    subprocess.run(['venv/bin/pip', 'install', '-r', 'requirements.txt'])
    print("Requirements installed.")

def main():
    """Main function to set up the environment."""
    setup_virtual_environment()
    install_requirements()

if __name__ == "__main__":
    main()