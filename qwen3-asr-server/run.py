import io
import logging
import torch
import tempfile
from pathlib import Path
from qwen_asr import Qwen3ASRModel
from fastapi import FastAPI, UploadFile, File, Form
import os
from typing import Optional


class TranscriptResult:
    def __init__(self, text: str, language: str):
        self.text = text
        self.language = language


class Qwen35Transcript():
    """Implementation based on qwen 3.5. """
    def __init__(self, model_name = "Qwen/Qwen3-ASR-1.7B"):
        """_summary_

        Args:
            model_name (str, optional): Qwen/Qwen3-ASR-1.7B or Qwen/Qwen3-ASR-0.6B
        """
        self._model_name = model_name
        # Load model on CPU
        self._model = Qwen3ASRModel.from_pretrained(
            self._model_name,
            device_map="cpu",               # CPU only
            dtype=torch.float32,        # use full precision on CPU
        )

    def transcribe(self, audio_data) -> TranscriptResult:

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "output.wav"
        
            with open(wav_path, "wb") as f:
                f.write(audio_data)
            audio_file = wav_path.as_posix()  # Convert Path to string for the model

            # Transcribe local audio file
            results = self._model.transcribe(
                audio=audio_file,  # local file path
                language=None,             # auto language detection
                return_time_stamps=False,  # set True if you want timestamps
            )

            # The results list contains objects with attributes `.text`, `.language`, etc.
            logging.debug(f"Transcript: {results[0].text}")
            logging.debug(f"Detected language: {results[0].language}")

        text = ""
        for segment in results:
            text += segment.text.strip() + "\n"
        text = text.strip()
        language = results[0].language
        
        # Supported languages according to https://github.com/QwenLM/Qwen3-ASR/blob/main/README.md
        # Chinese (zh), English (en), Cantonese (yue), Arabic (ar), German (de), French (fr), 
        # Spanish (es), Portuguese (pt), Indonesian (id), Italian (it), Korean (ko), Russian (ru), 
        # Thai (th), Vietnamese (vi), Japanese (ja), Turkish (tr), Hindi (hi), Malay (ms), Dutch (nl), 
        # Swedish (sv), Danish (da), Finnish (fi), Polish (pl), Czech (cs), Filipino (fil), Persian (fa), 
        # Greek (el), Hungarian (hu), Macedonian (mk), Romanian (ro)
        
        # mapping to ISO 639-1 codes
        language_mapping = {
            "Chinese": "zh",
            "English": "en",
            "Cantonese": "yue",
            "Arabic": "ar",
            "German": "de",
            "French": "fr",
            "Spanish": "es",
            "Portuguese": "pt",
            "Indonesian": "id",
            "Italian": "it",
            "Korean": "ko",
            "Russian": "ru",
            "Thai": "th",
            "Vietnamese": "vi",
            "Japanese": "ja",
            "Turkish": "tr",
            "Hindi": "hi",
            "Malay": "ms",
            "Dutch": "nl",
            "Swedish": "sv",
            "Danish": "da",
            "Finnish": "fi",
            "Polish": "pl",
            "Czech": "cs",
            "Filipino": "fil",
            "Persian": "fa",
            "Greek": "el",
            "Hungarian": "hu",
            "Macedonian": "mk",
            "Romanian": "ro"
        }
        if "," in language:
            language = language.split(",")[0].strip()  # Take the first language if multiple are detected
        language = language_mapping.get(language, "unknown")

        return TranscriptResult(text, language)


# FastAPI app
app = FastAPI(title="ASR Server", description="Audio Speech Recognition Server compatible with OpenAI API using Qwen3")

# Initialize transcriber (lazy load)
qwen_transcriber = None

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),  # Accept but ignore for compatibility
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(None)
):
    """
    Transcribe audio using Qwen3 ASR model, compatible with OpenAI API.
    
    Parameters:
    - file: Audio file to transcribe
    - model: Model to use (ignored, always uses Qwen3)
    - language: Language code (optional)
    - response_format: json, text, srt, verbose_json, vtt
    - temperature: Sampling temperature (ignored)
    """
    global qwen_transcriber
    if qwen_transcriber is None:
        qwen_transcriber = Qwen35Transcript()
    
    # Read the uploaded file
    audio_data = await file.read()
    
    # Transcribe using Qwen3
    result = qwen_transcriber.transcribe(audio_data)
    
    # Format response based on response_format
    if response_format == "json":
        return {"text": result.text}
    elif response_format == "text":
        return result.text
    elif response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": result.language,
            "duration": None,  # Qwen3 doesn't provide duration
            "text": result.text
        }
    elif response_format == "srt":
        # Simple SRT format (no timestamps from Qwen3)
        return "1\n00:00:00,000 --> 00:00:10,000\n" + result.text + "\n\n"
    elif response_format == "vtt":
        # Simple VTT format
        return "WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n" + result.text + "\n"
    else:
        return {"text": result.text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
