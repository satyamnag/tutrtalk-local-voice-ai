"""
OpenAI-compatible STT server using faster-whisper.
Replaces the vox-box binary, avoiding dependency conflicts.
"""

import argparse
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logger = logging.getLogger("whisper-stt")
logging.basicConfig(level=logging.INFO)

MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")  # tiny, base, small, medium, large
MODEL_ID = os.getenv("WHISPER_MODEL_ID", "whisper-1")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 for CPU, float16 for GPU

model = None


def load_model():
    global model
    logger.info("Loading faster-whisper model '%s' on %s...", MODEL_SIZE, DEVICE)
    from faster_whisper import WhisperModel

    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info("Model loaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Whisper STT Server", lifespan=lifespan)


def load_audio(audio_bytes: bytes, filename: str) -> np.ndarray:
    """Read audio bytes, resample to 16kHz mono float32."""
    suffix = os.path.splitext(filename)[1] if filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio, sample_rate = sf.read(tmp_path, dtype="float32")
    except Exception:
        waveform, sample_rate = torchaudio.load(tmp_path)
        audio = waveform.numpy()
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
    finally:
        os.unlink(tmp_path)

    if audio.ndim > 1:
        # Convert to mono
        audio = audio.mean(axis=1) if audio.shape[1] < audio.shape[0] else audio.mean(axis=0)

    if sample_rate != 16000:
        waveform = torch.tensor(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        audio = waveform.squeeze(0).numpy()

    return audio.astype(np.float32)


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    req_model: str = Form(MODEL_ID, alias="model"),
    response_format: Optional[str] = Form("json"),
    language: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
):
    # Use the global 'model' variable, not the request field
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        audio = load_audio(audio_bytes, file.filename or "audio.wav")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {e}")

    segments, _ = model.transcribe(audio, beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments)

    if response_format == "text":
        return PlainTextResponse(content=text)
    if response_format == "verbose_json":
        return JSONResponse(
            content={
                "text": text,
                "task": "transcribe",
                "language": "en",
                "duration": None,
            }
        )
    return JSONResponse(content={"text": text})


@app.get("/v1/models")
async def list_models():
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "openai",
                }
            ],
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper STT Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)