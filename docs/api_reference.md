# API Reference

## Overview
This document serves as a reference for the API endpoints available in the application. It includes details about the request methods, parameters, and response formats.

## Base URL
```
http://<host>:<port>/api
```

## Endpoints

### 1. Audio Capture
- **Endpoint:** `/audio/capture`
- **Method:** `POST`
- **Description:** Captures audio from the microphone.
- **Request Body:**
  - `duration` (int): Duration of the audio capture in seconds.
- **Response:**
  - `status` (string): Status of the capture (e.g., "success", "error").
  - `file_path` (string): Path to the saved audio file.

### 2. Speech to Text
- **Endpoint:** `/audio/speech_to_text`
- **Method:** `POST`
- **Description:** Converts captured audio to text.
- **Request Body:**
  - `file_path` (string): Path to the audio file.
- **Response:**
  - `transcription` (string): The transcribed text from the audio.

### 3. Sentiment Analysis
- **Endpoint:** `/analysis/sentiment`
- **Method:** `POST`
- **Description:** Analyzes the sentiment of the provided text.
- **Request Body:**
  - `text` (string): The text to analyze.
- **Response:**
  - `sentiment` (string): The sentiment result (e.g., "positive", "negative", "neutral").

### 4. Profanity Detection
- **Endpoint:** `/analysis/profanity`
- **Method:** `POST`
- **Description:** Detects profanity in the provided text.
- **Request Body:**
  - `text` (string): The text to check for profanity.
- **Response:**
  - `contains_profanity` (boolean): Indicates if profanity was detected.

### 5. Emotion Analysis
- **Endpoint:** `/analysis/emotion`
- **Method:** `POST`
- **Description:** Analyzes emotions expressed in the provided text.
- **Request Body:**
  - `text` (string): The text to analyze.
- **Response:**
  - `emotions` (object): A breakdown of detected emotions (e.g., happiness, sadness).

### 6. Stress Detection
- **Endpoint:** `/analysis/stress`
- **Method:** `POST`
- **Description:** Detects stress levels based on audio or text input.
- **Request Body:**
  - `input_type` (string): Type of input ("audio" or "text").
  - `data` (string): The audio file path or text to analyze.
- **Response:**
  - `stress_level` (string): The detected stress level (e.g., "low", "medium", "high").

### 7. LLM Analysis
- **Endpoint:** `/analysis/llm`
- **Method:** `POST`
- **Description:** Analyzes data using large language models.
- **Request Body:**
  - `text` (string): The text to analyze.
- **Response:**
  - `analysis_result` (object): The result of the LLM analysis.

## Error Handling
All endpoints return appropriate HTTP status codes and error messages in case of failures. Common status codes include:
- `200 OK`: Successful request.
- `400 Bad Request`: Invalid input.
- `404 Not Found`: Endpoint not found.
- `500 Internal Server Error`: Server error.

## Authentication
Some endpoints may require authentication. Please refer to the authentication section in the documentation for details on how to obtain and use API tokens.

## Conclusion
This API reference provides a comprehensive overview of the available endpoints and their functionalities. For further assistance, please refer to the user manual or contact support.