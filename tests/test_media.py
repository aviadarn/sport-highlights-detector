import subprocess
import pytest
from nba_highlights.media import (
    classify_source, ingest, probe_duration, frame_times, extract_frame,
)
from nba_highlights.errors import IngestError


def test_classify_source():
    assert classify_source("https://www.youtube.com/watch?v=x") == "youtube"
    assert classify_source("https://youtu.be/x") == "youtube"
    assert classify_source("https://cdn.example.com/stream.m3u8") == "hls"
    assert classify_source("/tmp/game.mp4") == "file"


def test_frame_times_evenly_spaced():
    assert frame_times(0.0, 10.0, 1) == [5.0]
    assert frame_times(0.0, 12.0, 3) == [3.0, 6.0, 9.0]
    assert frame_times(4.0, 4.0, 2) == [4.0, 4.0]  # zero-span is safe


def test_ingest_file_runs_ffmpeg_for_audio(tmp_path):
    calls = []

    def fake_runner(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    video, audio = ingest("/videos/game.mp4", str(tmp_path), runner=fake_runner)
    assert video == "/videos/game.mp4"
    assert audio.endswith("audio.wav")
    assert calls[0][0] == "ffmpeg"
    assert "16000" in calls[0]


def test_ingest_download_failure_raises_ingest_error(tmp_path):
    def boom(url, out_template):
        raise RuntimeError("network down")

    with pytest.raises(IngestError):
        ingest("https://youtu.be/x", str(tmp_path), downloader=boom)


def test_extract_frame_invokes_ffmpeg(tmp_path):
    calls = []
    out = str(tmp_path / "f.jpg")
    extract_frame("/v.mp4", 12.5, out,
                  runner=lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    assert calls[0][0] == "ffmpeg" and out in calls[0]


def test_probe_duration_parses_and_defaults():
    # Happy path: parses stdout as float
    def fake_runner(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="40.0\n")

    assert probe_duration("v.mp4", runner=fake_runner) == 40.0

    # Error path: any exception returns 0.0
    def boom(cmd, **kw):
        raise RuntimeError("ffprobe failed")

    assert probe_duration("v.mp4", runner=boom) == 0.0
