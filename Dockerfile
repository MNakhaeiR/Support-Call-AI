# runpod Dockerfile for running an AI model
FROM python:3.11-slim-buster
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
# Copy the application code
COPY . /app
# Set the working directory
WORKDIR /app
# Expose the port for the application
EXPOSE 8000
# Command to run the application
CMD ["python", "app.py"]
