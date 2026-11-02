"""
Savoryne recipe video service (Render).

POST /transcribe  {"url": "...", "api_key": "...", "frames": 8}
  -> {"title", "caption", "transcript", "has_speech", "duration", "frames": [base64 jpeg, ...]}

Downloads the video with yt-dlp, transcribes the audio (if any) with OpenAI,
grabs evenly spaced snapshots with ffmpeg, and returns the creator's caption,
so the recipe can be drafted even when nobody talks in the video.

Env vars:
  OPENAI_API_KEY        required
  TRANSCRIBER_API_KEY   shared secret sent by WordPress (api_key)
  TRANSCRIBE_MODEL      optional, default "whisper-1"
  MAX_FRAMES            optional, default 8
"""

import base64
import glob
import os
import re
import subprocess
import tempfile

import static_ffmpeg
import yt_dlp
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

# Puts ffmpeg + ffprobe on PATH (downloaded at build time; system ffmpeg wins if present).
static_ffmpeg.add_paths(weak=True)

API_KEY = os.environ.get("TRANSCRIBER_API_KEY", "")
TRANSCRIBE_MODEL = os.environ.get("TRANSCRIBE_MODEL", "whisper-1")
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "8"))
MAX_AUDIO_SECONDS = 600
MAX_VIDEO_SECONDS = int(os.environ.get("MAX_VIDEO_SECONDS", "420"))  # 7 minutes
ANSI = re.compile(r"\x1b\[[0-9;]*m")

client = OpenAI()
app = FastAPI(title="Savoryne video service")


class VideoTooLong(Exception):
    pass


class TranscribeRequest(BaseModel):
    url: str
    api_key: str = ""
    frames: int | None = None


def run(cmd, timeout=120):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def friendly_error(err) -> str:
    msg = ANSI.sub("", str(err)).lower()
    if "unsupported url" in msg:
        return "That link isn’t a supported video. Paste a TikTok, Instagram, Facebook, Vimeo or X link."
    if any(w in msg for w in ("private", "login", "log in", "sign in", "cookies")):
        return "That video is private or needs a login, so it can’t be read."
    if any(w in msg for w in ("404", "not found", "unavailable", "removed", "deleted")):
        return "That video is unavailable. It may have been removed."
    return "That video couldn’t be downloaded. Check the link and try again."


def download(url: str, folder: str):
    opts = {
        "outtmpl": os.path.join(folder, "video.%(ext)s"),
        "format": "b[height<=720]/bv*[height<=720]+ba/b/bv*+ba",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 2,
        "max_filesize": 250 * 1024 * 1024,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info.get("entries"):
            info = info["entries"][0] or {}
        duration = float(info.get("duration") or 0)
        if duration > MAX_VIDEO_SECONDS:
            raise VideoTooLong(f"That video is {duration / 60:.1f} minutes long. The limit is {MAX_VIDEO_SECONDS // 60} minutes.")
        info = ydl.process_ie_result(info, download=True)
    files = [f for f in glob.glob(os.path.join(folder, "video.*")) if not f.endswith((".part", ".ytdl"))]
    if not files:
        raise RuntimeError("Download produced no file.")
    return info, files[0]


def has_stream(path: str, kind: str) -> bool:
    r = run(["ffprobe", "-v", "error", "-select_streams", kind, "-show_entries", "stream=index", "-of", "csv=p=0", path], 30)
    return r.returncode == 0 and r.stdout.strip() != ""


def probe_duration(path: str) -> float:
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path], 30)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def extract_audio(video: str, folder: str):
    out = os.path.join(folder, "audio.mp3")
    r = run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000",
             "-b:a", "48k", "-t", str(MAX_AUDIO_SECONDS), out], 180)
    if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1000:
        return out
    return None


def transcribe_audio(path: str) -> str:
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=f)
    return (getattr(result, "text", "") or "").strip()


def extract_frames(video: str, duration: float, count: int, folder: str):
    if count <= 0:
        return []
    n = count if duration > 4 else 1
    frames = []
    for i in range(n):
        t = duration * (i + 0.5) / n if duration > 0 else 0
        out = os.path.join(folder, f"frame{i}.jpg")
        r = run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
                 "-vf", "scale='min(768,iw)':-2", "-q:v", "5", out], 60)
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
            with open(out, "rb") as f:
                frames.append(base64.b64encode(f.read()).decode("ascii"))
    return frames


@app.get("/")
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    if API_KEY and req.api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Paste a full video link that starts with https://")

    count = max(0, min(12, req.frames if req.frames is not None else MAX_FRAMES))

    with tempfile.TemporaryDirectory() as tmp:
        try:
            info, video = download(url, tmp)
        except VideoTooLong as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:  # yt-dlp raises several error types
            print("Download failed:", ANSI.sub("", str(e)), flush=True)
            raise HTTPException(status_code=400, detail=friendly_error(e))

        duration = float(info.get("duration") or 0) or probe_duration(video)

        transcript = ""
        if has_stream(video, "a"):
            audio = extract_audio(video, tmp)
            if audio:
                try:
                    transcript = transcribe_audio(audio)
                except Exception as e:
                    print("Transcription failed:", e, flush=True)

        frames = extract_frames(video, duration, count, tmp) if has_stream(video, "v") else []

    caption = (info.get("description") or "").strip()
    title = (info.get("title") or "").strip()

    if not transcript and not caption and not frames:
        raise HTTPException(status_code=422, detail="Nothing usable was found in that video. Try a different link.")

    return {
        "title": title,
        "caption": caption,
        "transcript": transcript,
        "has_speech": len(transcript.split()) >= 8,
        "duration": duration,
        "frames": frames,
    }
