import os

def read_file(file_path):
    """Read the contents of a file and return it."""
    with open(file_path, 'r') as file:
        return file.read()

def write_file(file_path, data):
    """Write data to a file."""
    with open(file_path, 'w') as file:
        file.write(data)

def delete_file(file_path):
    """Delete a file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)

def list_files(directory):
    """List all files in a directory."""
    return os.listdir(directory)

def create_directory(directory_path):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)