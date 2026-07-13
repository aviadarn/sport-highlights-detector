import os
import subprocess
from urllib.parse import urlparse
from nba_highlights.errors import IngestError


def classify_source(locator: str) -> str:
    host = (urlparse(locator).netloc or "").lower()
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if locator.lower().endswith(".m3u8"):
        return "hls"
    return "file"


def _default_downloader(url: str, out_template: str) -> str:
    from yt_dlp import YoutubeDL
    opts = {"format": "mp4/best", "outtmpl": out_template, "quiet": True,
            "noplaylist": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def ingest(locator: str, workdir: str, downloader=_default_downloader,
           runner=subprocess.run) -> tuple[str, str]:
    os.makedirs(workdir, exist_ok=True)
    kind = classify_source(locator)
    try:
        if kind == "file":
            video_path = locator
        else:
            video_path = downloader(locator, os.path.join(workdir, "video.%(ext)s"))
        audio_path = os.path.join(workdir, "audio.wav")
        runner(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
                "-ar", "16000", audio_path], check=True)
    except IngestError:
        raise
    except Exception as e:  # download/ffmpeg failure -> uniform error
        raise IngestError(f"ingest failed for {locator!r}: {e}") from e
    return video_path, audio_path


def probe_duration(video_path: str, runner=subprocess.run) -> float:
    try:
        result = runner(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def frame_times(start_s: float, end_s: float, n: int) -> list[float]:
    n = max(1, n)
    span = max(0.0, end_s - start_s)
    return [start_s + (j + 1) / (n + 1) * span for j in range(n)]


def extract_frame(video_path: str, t_s: float, out_path: str,
                  runner=subprocess.run) -> str:
    runner(["ffmpeg", "-y", "-ss", str(t_s), "-i", video_path,
            "-frames:v", "1", "-q:v", "2", out_path], check=True)
    return out_path
