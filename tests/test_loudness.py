import numpy as np
from nba_highlights.audio.loudness import rms_envelope, shot_loudness


def test_rms_envelope_shapes_align():
    sr = 1000
    samples = np.ones(sr * 2, dtype=float)  # 2 seconds of constant signal
    times, rms = rms_envelope(samples, sr, hop_s=0.1, win_s=0.1)
    assert len(times) == len(rms)
    assert np.allclose(rms, 1.0, atol=1e-6)


def test_shot_loudness_finds_peak_in_window():
    sr = 1000
    samples = np.zeros(sr * 3, dtype=float)
    samples[sr * 1 : sr * 1 + 100] = 5.0  # loud burst at ~1.0s
    times, rms = rms_envelope(samples, sr, hop_s=0.05, win_s=0.05)
    peak, t_peak = shot_loudness(times, rms, 0.5, 1.5)
    assert peak > 0.0
    assert 0.9 <= t_peak <= 1.2


def test_shot_loudness_empty_window():
    times = np.array([0.0, 0.1, 0.2])
    rms = np.array([1.0, 1.0, 1.0])
    assert shot_loudness(times, rms, 5.0, 6.0) == (0.0, None)
