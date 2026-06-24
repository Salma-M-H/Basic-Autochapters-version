#!/usr/bin/env python3
"""
transcriber.py  —  Step 1 of 3
================================
Extracts audio from a video file and transcribes it using Groq Whisper.
Audio is split into equal fixed-duration chunks (no silence detection).
Chunks are transcribed in parallel using ThreadPoolExecutor.

Output:
    transcript.txt   — every line is:  [HH:MM:SS] spoken text

Requirements:
    pip install groq moviepy pydub python-dotenv
"""

import os
import sys
import time
import tempfile
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ============================================================
#  CONFIG
# ============================================================

VIDEO_PATH   = "video.mp4"
OUTPUT_FILE  = "transcript.txt"

FFMPEG_PATH  = os.getenv("FFMPEG_PATH",  "ffmpeg")
FFPROBE_PATH = os.getenv("FFPROBE_PATH", "ffprobe")

MAX_RETRIES      = 5     # retries per chunk on connection error
RETRY_DELAY      = 5     # seconds before first retry (doubles each attempt)
CHUNK_MINUTES    = 10    # each chunk is this many minutes long
MAX_WORKERS      = 3     # parallel Groq Whisper requests

# ============================================================


# ── Patch pydub to use the configured ffprobe ──────────────
import pydub.utils
import pydub.audio_segment
import json

def _ffprobe_mediainfo_json(filename, read_ahead_limit=-1):
    cmd = [FFPROBE_PATH, "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(filename)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not result.stdout.strip():
        raise RuntimeError(
            f"ffprobe returned no output for '{filename}'.\nstderr: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)

pydub.utils.mediainfo_json         = _ffprobe_mediainfo_json
pydub.audio_segment.mediainfo_json = _ffprobe_mediainfo_json

from pydub import AudioSegment

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe   = FFPROBE_PATH
# ───────────────────────────────────────────────────────────


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_audio(video_path: str, output_path: str) -> None:
    try:
        from moviepy import VideoFileClip
    except ImportError:
        print("Error: moviepy not installed. Run: pip install moviepy")
        sys.exit(1)

    print(f"[1/2] Extracting audio from: {video_path}")
    clip  = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_path, logger=None)
    audio.close()
    clip.close()
    print(f"      Saved to: {output_path}")


def chunk_audio(audio_path: str) -> list[tuple[AudioSegment, float]]:
    """
    Split audio into equal fixed-duration chunks of CHUNK_MINUTES each.
    Returns a list of (chunk_audio, start_seconds) tuples.
    """
    audio        = AudioSegment.from_mp3(audio_path)
    total_ms     = len(audio)
    chunk_ms     = CHUNK_MINUTES * 60 * 1000

    if total_ms <= chunk_ms:
        print(f"      Audio is short — sending as one chunk.")
        return [(audio, 0.0)]

    chunks = []
    pos_ms = 0
    while pos_ms < total_ms:
        end_ms = min(pos_ms + chunk_ms, total_ms)
        chunks.append((audio[pos_ms:end_ms], pos_ms / 1000))
        pos_ms = end_ms

    print(f"      Split into {len(chunks)} chunks of ~{CHUNK_MINUTES} min each.")
    return chunks


def transcribe_chunk_with_retry(chunk_path: str, client) -> object:
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(chunk_path, "rb") as f:
                return client.audio.transcriptions.create(
                    file=(Path(chunk_path).name, f),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    language="ar",
                )
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"\n      [ERROR] Failed after {MAX_RETRIES} attempts: {type(e).__name__}: {e}")
                raise
            print(f"\n      [{type(e).__name__}] Attempt {attempt}/{MAX_RETRIES} failed. "
                  f"Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def _process_chunk(args: tuple) -> tuple[int, object, float]:
    """
    Worker: exports one chunk to a temp MP3, calls Groq Whisper,
    returns (index, transcription, start_seconds).
    """
    i, chunk, start_seconds, client, tmp_dir = args
    chunk_path = os.path.join(tmp_dir, f"chunk_{i}.mp3")
    chunk.export(chunk_path, format="mp3")
    print(f"      Chunk {i + 1}: starts at {format_timestamp(start_seconds)}, "
          f"{len(chunk) / 1000:.0f}s — transcribing...", flush=True)
    transcription = transcribe_chunk_with_retry(chunk_path, client)
    print(f"      Chunk {i + 1}: done.")
    return i, transcription, start_seconds


def transcribe(audio_path: str, client) -> str:
    """
    Transcribe the audio and return the full transcript as a string.
    Each line: [HH:MM:SS] spoken text
    """
    print("[2/2] Transcribing with Groq Whisper...")

    audio_chunks = chunk_audio(audio_path)
    total        = len(audio_chunks)
    results      = {}
    tmp_dir      = tempfile.mkdtemp()

    try:
        if total == 1:
            chunk, start_seconds = audio_chunks[0]
            _, transcription, start_seconds = _process_chunk(
                (0, chunk, start_seconds, client, tmp_dir)
            )
            results[0] = (transcription, start_seconds)
        else:
            print(f"      Transcribing {total} chunks in parallel (max {MAX_WORKERS} workers)...")
            task_args = [
                (i, chunk, start_s, client, tmp_dir)
                for i, (chunk, start_s) in enumerate(audio_chunks)
            ]
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(_process_chunk, args): args[0] for args in task_args}
                for future in as_completed(futures):
                    i, transcription, start_seconds = future.result()
                    results[i] = (transcription, start_seconds)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Reassemble in original order
    lines = []
    for i in sorted(results):
        transcription, start_seconds = results[i]
        for seg in transcription.segments:
            seg_start = seg["start"] if isinstance(seg, dict) else seg.start
            seg_text  = seg["text"]  if isinstance(seg, dict) else seg.text
            real_time = seg_start + start_seconds
            lines.append(f"[{format_timestamp(real_time)}] {seg_text.strip()}")

    print(f"      Transcription complete: {len(lines)} lines.")
    return "\n".join(lines)


def download_audio_from_youtube(url: str, output_path: str) -> None:
    """Download audio from a YouTube URL and save as MP3 using pytubefix."""
    try:
        from pytubefix import YouTube
    except ImportError:
        raise RuntimeError("pytubefix not installed. Run: pip install pytubefix")

    print(f"[1/2] Downloading audio from YouTube: {url}")

    yt     = YouTube(url)
    stream = yt.streams.get_audio_only()
    if not stream:
        raise RuntimeError("No audio stream found for this YouTube URL.")

    # pytubefix downloads to a directory; we control filename via output_path
    out_dir      = str(Path(output_path).parent)
    tmp_filename = "yt_audio_tmp"
    downloaded   = stream.download(output_path=out_dir, filename=tmp_filename)

    # Convert to MP3 using ffmpeg (pytubefix downloads as .mp4/webm)
    result = subprocess.run(
        [FFMPEG_PATH, "-y", "-i", downloaded, "-vn",
         "-ar", "44100", "-ac", "2", "-b:a", "192k", output_path],
        capture_output=True, text=True,
    )
    Path(downloaded).unlink(missing_ok=True)   # remove the raw download

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr.strip()}")

    print(f"      Saved to: {output_path}")


def transcribe_from_video(video_path: str, client) -> str:
    """Extract audio from a local video file and transcribe it."""
    tmp_dir = tempfile.mkdtemp()
    try:
        audio_path = os.path.join(tmp_dir, "audio.mp3")
        extract_audio(video_path, audio_path)
        return transcribe(audio_path, client)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def transcribe_from_youtube(url: str, client) -> str:
    """Download audio from a YouTube URL and transcribe it."""
    tmp_dir = tempfile.mkdtemp()
    try:
        audio_path = os.path.join(tmp_dir, "audio.mp3")
        download_audio_from_youtube(url, audio_path)
        return transcribe(audio_path, client)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in .env")
        sys.exit(1)
    if not os.path.isfile(VIDEO_PATH):
        print(f"Error: Video not found: {VIDEO_PATH}")
        sys.exit(1)

    try:
        from groq import Groq
    except ImportError:
        print("Error: groq not installed. Run: pip install groq")
        sys.exit(1)

    client  = Groq(api_key=groq_api_key)
    tmp_dir = tempfile.mkdtemp()

    print("\n=== Step 1: Transcription ===\n")
    try:
        audio_path = os.path.join(tmp_dir, "audio.mp3")
        extract_audio(VIDEO_PATH, audio_path)
        transcript = transcribe(audio_path, client)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    output_path = Path(__file__).parent / OUTPUT_FILE
    output_path.write_text(transcript, encoding="utf-8")
    print(f"\n  Transcript saved → {output_path}")
    print(f"\n--- Preview (first 5 lines) ---")
    print("\n".join(transcript.split("\n")[:5]))
    print(f"\n=== Done. Run segmenter.py next. ===\n")


if __name__ == "__main__":
    main()