import pytest
from nba_highlights.config import Settings, Weights
from nba_highlights.errors import ConfigError


def test_defaults():
    s = Settings()
    assert s.action_backend == "stub"
    assert s.asr_backend == "stub"
    assert s.threshold == 0.5
    assert s.audio_gate == 0.35
    assert s.max_gap_s == 3.0
    assert "dunk" in s.hype_lexicon


def test_validate_rejects_unknown_backend():
    with pytest.raises(ConfigError):
        Settings(action_backend="nope").check()
    with pytest.raises(ConfigError):
        Settings(asr_backend="nope").check()


def test_validate_rejects_bad_weight_sum():
    with pytest.raises(ConfigError):
        Settings(weights=Weights(wL=0.5, wP=0.5, wK=0.5)).check()


def test_check_rejects_bad_fusion_weight_sum():
    with pytest.raises(ConfigError):
        Settings(weights=Weights(wA=0.9, wV=0.9)).check()


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("ACTION_BACKEND", "videomae")
    monkeypatch.setenv("ASR_BACKEND", "faster-whisper")
    monkeypatch.setenv("THRESHOLD", "0.7")
    s = Settings.from_env()
    assert s.action_backend == "videomae"
    assert s.asr_backend == "faster-whisper"
    assert s.threshold == 0.7
