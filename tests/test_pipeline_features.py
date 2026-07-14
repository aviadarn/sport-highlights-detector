import subprocess
import numpy as np
from nba_highlights.pipeline import run_pipeline
from nba_highlights.config import Settings
from nba_highlights.transcribe.stub import StubTranscriber
from nba_highlights.action.stub import StubActionRecognizer
from nba_highlights.scoring.stub import StubScorer
from nba_highlights.features import ShotFeatures


def _runner(cmd, **kw):
    if cmd and cmd[0] == "ffprobe":
        return subprocess.CompletedProcess(cmd, 0, stdout="40.0\n")
    return subprocess.CompletedProcess(cmd, 0)


def _audio_loader(path):
    sr = 1000
    samples = np.zeros(sr * 40, dtype=float)
    samples[int(sr * 2.7):int(sr * 3.9)] = 3.0
    return samples, sr


def _detector(path):
    return [(0.0, 2.0), (2.5, 4.0), (20.0, 22.0)]


def test_emit_features_writes_one_row_per_shot(tmp_path):
    feats_path = tmp_path / "feats.jsonl"
    run_pipeline(
        "/v.mp4", Settings(threshold=0.4, audio_gate=0.2), StubTranscriber(),
        StubActionRecognizer(scores={2: 0.9}),
        workdir=str(tmp_path), generated_at="t",
        runner=_runner, detector_fn=_detector, audio_loader=_audio_loader,
        emit_features_path=str(feats_path))
    rows = [ShotFeatures.model_validate_json(ln)
            for ln in feats_path.read_text().splitlines() if ln.strip()]
    assert len(rows) == 3                      # one per detected shot
    assert rows[1].scene_id == 1
    assert 0.0 <= rows[1].rel_position <= 1.0
    assert rows[1].shot_duration_s == 1.5      # 4.0 - 2.5


def test_injected_scorer_drives_selection(tmp_path):
    # StubScorer marks only scene 2 hot; it must be the sole highlight.
    report = run_pipeline(
        "/v.mp4", Settings(threshold=0.5, audio_gate=0.0), StubTranscriber(),
        StubActionRecognizer(),
        workdir=str(tmp_path), generated_at="t",
        runner=_runner, detector_fn=_detector, audio_loader=_audio_loader,
        scorer=StubScorer(scores={2: 0.95}, default=0.0))
    assert len(report.highlights) == 1
    assert report.highlights[0].member_scenes == [2]
    assert report.config["fusion"] == "weighted"


def test_weighted_default_is_unchanged(tmp_path):
    # Regression: default (no scorer) still finds the loud+hype scene.
    report = run_pipeline(
        "/v.mp4", Settings(threshold=0.4, audio_gate=0.2), StubTranscriber(),
        StubActionRecognizer(scores={2: 0.9}),
        workdir=str(tmp_path), generated_at="t",
        runner=_runner, detector_fn=_detector, audio_loader=_audio_loader)
    assert len(report.highlights) == 1
