import subprocess
from nba_highlights.pipeline import run_pipeline
from nba_highlights.config import Settings, Weights
from nba_highlights.transcribe.stub import StubTranscriber
from nba_highlights.action.stub import StubActionRecognizer
from nba_highlights.interfaces import Transcript, TranscriptSegment, Word
import numpy as np


def _fake_runner(cmd, **kw):
    if cmd and cmd[0] == "ffprobe":
        return subprocess.CompletedProcess(cmd, 0, stdout="40.0\n")
    return subprocess.CompletedProcess(cmd, 0)


def _transcript():
    # loud hype words land inside scene 1 (2.5-4.0)
    return Transcript(segments=[TranscriptSegment(
        start_s=2.6, end_s=3.9, text="what a dunk and one",
        words=[Word(text="what", start_s=2.6, end_s=2.8),
               Word(text="a", start_s=2.8, end_s=2.9),
               Word(text="dunk", start_s=3.0, end_s=3.3),
               Word(text="and", start_s=3.4, end_s=3.6),
               Word(text="one", start_s=3.6, end_s=3.9)])])


def _audio_loader(path):
    sr = 1000
    samples = np.zeros(sr * 40, dtype=float)
    samples[int(sr * 2.7):int(sr * 3.9)] = 3.0  # loud burst in scene 1
    return samples, sr


def _detector(path):
    return [(0.0, 2.0), (2.5, 4.0), (20.0, 22.0)]


def test_pipeline_detects_the_loud_hype_scene(tmp_path):
    settings = Settings(threshold=0.4, audio_gate=0.2, weights=Weights())
    action = StubActionRecognizer(scores={2: 0.9}, default=0.0)  # scene starting at 2.5
    report = run_pipeline(
        "/videos/game.mp4", settings,
        transcriber=StubTranscriber(_transcript()),
        action=action,
        workdir=str(tmp_path), generated_at="2026-07-13T00:00:00Z",
        title="Game 7",
        runner=_fake_runner, detector_fn=_detector, audio_loader=_audio_loader,
    )
    assert report.video.title == "Game 7"
    assert report.video.duration_s == 40.0
    assert len(report.highlights) == 1
    h = report.highlights[0]
    assert h.start_s == 2.5 and h.end_s == 4.0
    assert "dunk" in h.evidence.keywords and "and one" in h.evidence.keywords
    assert report.config["action_backend"] == "stub"


def test_pipeline_bad_config_raises(tmp_path):
    from nba_highlights.errors import ConfigError
    import pytest
    settings = Settings(weights=Weights(wA=0.9, wV=0.9))
    with pytest.raises(ConfigError):
        run_pipeline("/v.mp4", settings, StubTranscriber(), StubActionRecognizer(),
                     workdir=str(tmp_path), generated_at="t",
                     runner=_fake_runner, detector_fn=_detector, audio_loader=_audio_loader)
