import numpy as np
from nba_highlights.audio.prosody import shot_prosody


def test_prosody_zero_for_silence():
    sr = 1000
    samples = np.zeros(sr * 2, dtype=float)
    assert shot_prosody(samples, sr, 0.0, 2.0) == 0.0


def test_prosody_higher_for_loud_high_freq():
    sr = 2000
    t = np.arange(sr * 2) / sr
    calm = 0.1 * np.sin(2 * np.pi * 100 * t)      # quiet, low freq
    excited = 1.0 * np.sin(2 * np.pi * 500 * t)   # loud, high freq
    calm_score = shot_prosody(calm, sr, 0.0, 2.0)
    excited_score = shot_prosody(excited, sr, 0.0, 2.0)
    assert excited_score > calm_score


def test_prosody_empty_window_is_zero():
    sr = 1000
    samples = np.ones(sr, dtype=float)
    assert shot_prosody(samples, sr, 5.0, 6.0) == 0.0
