# qwen3-asr-server

A web server providing audio speech recognition compatible with OpenAI's API, using Qwen3 ASR models.

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI's transcription API
- **Qwen3 ASR Backend**: Uses Qwen3 models for local transcription
- FastAPI-based web server
- Support for multiple response formats (json, text, srt, vtt, verbose_json)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server:
   ```bash
   python qwen3-asr-server/run.py
   ```

The server will start on `http://localhost:8000`.

## API Endpoints

### POST /v1/audio/transcriptions

Transcribe audio using Qwen3 ASR model, fully compatible with OpenAI's API.

**Parameters:**
- `file` (required): Audio file to transcribe (multipart/form-data)
- `model` (optional): Model name (ignored, always uses Qwen3)
- `language` (optional): Language code for transcription
- `response_format` (optional): `json`, `text`, `srt`, `vtt`, `verbose_json` (default: `json`)
- `temperature` (optional): Sampling temperature (ignored)

**Response formats:**

- `json`: `{"text": "transcribed text"}`
- `text`: Plain text response
- `verbose_json`: `{"task": "transcribe", "language": "en", "text": "..."}`
- `srt`: SRT subtitle format
- `vtt`: WebVTT subtitle format

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav" \
     -F "response_format=json"
```

## Requirements

- Python 3.8+
- Internet connection for model download (first run only)

## TODO
- Fix duration processing
- Make model selectable through endpoint