import shutil
import subprocess
import pytest

pytest.importorskip("scenedetect")
pytest.importorskip("soundfile")

pytestmark = pytest.mark.slow


def _make_clip(path):
    # 6s test video with a scene change at 3s (color flip) + a tone, via ffmpeg
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "color=c=red:s=320x240:d=3",
         "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-map", "2:a", "-shortest", path],
        check=True)


def test_pipeline_on_real_clip(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    from nba_highlights.pipeline import run_pipeline
    from nba_highlights.config import Settings
    from nba_highlights.transcribe.stub import StubTranscriber
    from nba_highlights.action.stub import StubActionRecognizer

    clip = str(tmp_path / "clip.mp4")
    _make_clip(clip)
    settings = Settings(threshold=0.0, audio_gate=0.0)  # accept everything
    report = run_pipeline(
        clip, settings, StubTranscriber(), StubActionRecognizer(default=0.5),
        workdir=str(tmp_path / "work"), generated_at="2026-07-13T00:00:00Z")
    assert report.video.duration_s > 0
    assert isinstance(report.highlights, list)
