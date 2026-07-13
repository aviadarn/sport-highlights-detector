from nba_highlights.fuse import percentile_normalize, audio_composite, fuse
from nba_highlights.config import Weights


def test_percentile_normalize_ranks():
    assert percentile_normalize([10.0, 20.0, 30.0]) == [0.0, 0.5, 1.0]


def test_percentile_normalize_all_equal():
    assert percentile_normalize([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_percentile_normalize_edge_cases():
    assert percentile_normalize([]) == []
    assert percentile_normalize([7.0]) == [0.0]


def test_percentile_normalize_preserves_index_order():
    # output[i] must correspond to values[i], not a sorted list
    assert percentile_normalize([30.0, 10.0, 20.0]) == [1.0, 0.0, 0.5]


def test_audio_composite_weighting():
    w = Weights(wL=0.4, wP=0.3, wK=0.3)
    assert abs(audio_composite(1.0, 0.0, 0.0, w) - 0.4) < 1e-9
    assert abs(audio_composite(1.0, 1.0, 1.0, w) - 1.0) < 1e-9


def test_fuse_weighting():
    w = Weights(wA=0.5, wV=0.5)
    assert abs(fuse(0.8, 0.4, w) - 0.6) < 1e-9
