import os
import tempfile
import subprocess
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class TranscribeRequest(BaseModel):
    url: str
    api_key: str  # WordPress sends this to verify requests

ALLOWED_API_KEY = os.environ.get("TRANSCRIBER_API_KEY", "")

@app.get("/")
def root():
    return {
        "service": "Video Transcription API",
        "status": "running",
        "endpoints": {
            "GET /health": "Health check",
            "POST /transcribe": "Transcribe video from URL (max 7 minutes)"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    # Verify the request is coming from your WordPress site
    if ALLOWED_API_KEY and req.api_key != ALLOWED_API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    # Validate it's a supported video URL
    supported = ["youtube.com", "youtu.be", "tiktok.com", "instagram.com", "facebook.com", "vimeo.com", "twitter.com", "x.com"]
    if not any(d in url for d in supported):
        raise HTTPException(status_code=400, detail="Unsupported video platform")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        # First, get video info to check duration
        info_cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-playlist",
            url
        ]
        
        try:
            info_result = subprocess.run(
                info_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if info_result.returncode == 0:
                import json
                video_info = json.loads(info_result.stdout)
                duration = video_info.get("duration", 0)
                
                # Check if video is longer than 7 minutes (420 seconds)
                if duration > 420:
                    minutes = duration / 60
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Video is too long ({minutes:.1f} minutes). Maximum allowed is 7 minutes."
                    )
        except json.JSONDecodeError:
            pass  # Continue if we can't get duration
        except subprocess.TimeoutExpired:
            pass  # Continue if info check times out

       # Download audio using yt-dlp with better bot avoidance
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "--extractor-args", "youtube:player_client=android,web",
            "--no-check-certificates",
            "-o", audio_path,
            url
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Video download timed out. Try a shorter video.")

        if result.returncode != 0:
            err = result.stderr[-500:] if result.stderr else "Unknown error"
            raise HTTPException(status_code=422, detail=f"Could not download video: {err}")

        # yt-dlp sometimes appends .mp3 to the output path
        if not os.path.exists(audio_path):
            audio_path = audio_path + ".mp3" if os.path.exists(audio_path + ".mp3") else None
            if not audio_path:
                raise HTTPException(status_code=422, detail="Audio file not found after download")

        # Check file size (Whisper limit is 25MB)
        file_size = os.path.getsize(audio_path)
        if file_size > 24 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio file too large (max ~24MB). Try a shorter video.")

        # Transcribe with Whisper
        try:
            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text"
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    return {"transcript": transcript}
