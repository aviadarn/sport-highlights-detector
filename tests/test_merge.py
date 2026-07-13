from nba_highlights.merge import merge_shots
from nba_highlights.models import ShotScore, AudioSignals, ActionSignals


def _ss(scene_id, start, end, fused, kw=None, loud=0.0, peak=None, txt=""):
    return ShotScore(
        scene_id=scene_id, start_s=start, end_s=end,
        audio=AudioSignals(loudness=loud), action=ActionSignals(score=fused, label="x"),
        audio_composite=fused, fused=fused, matched_keywords=kw or [],
        peak_loudness_s=peak, transcript=txt)


def test_merges_adjacent_within_gap():
    shots = [_ss(0, 0.0, 2.0, 0.8, kw=["dunk"], peak=1.0, txt="what a dunk"),
             _ss(1, 2.5, 4.0, 0.9, kw=["and one"], peak=3.0, txt="and one"),
             _ss(2, 30.0, 32.0, 0.7, kw=["three"], peak=31.0, txt="from three")]
    out = merge_shots(shots, threshold=0.5, max_gap_s=3.0)
    assert len(out) == 2
    assert out[0].member_scenes == [0, 1]
    assert out[0].start_s == 0.0 and out[0].end_s == 4.0
    assert out[0].score == 0.9              # max fused of the group
    assert out[0].evidence.keywords == ["dunk", "and one"]
    assert out[0].evidence.peak_loudness_s == 3.0   # from the peak member (scene 1)
    assert out[0].id == 0 and out[1].id == 1


def test_below_threshold_shots_excluded():
    shots = [_ss(0, 0.0, 2.0, 0.4), _ss(1, 2.5, 4.0, 0.9)]
    out = merge_shots(shots, threshold=0.5, max_gap_s=3.0)
    assert len(out) == 1 and out[0].member_scenes == [1]


def test_gap_too_large_splits():
    shots = [_ss(0, 0.0, 2.0, 0.8), _ss(1, 10.0, 12.0, 0.8)]
    out = merge_shots(shots, threshold=0.5, max_gap_s=3.0)
    assert len(out) == 2


def test_empty_input():
    assert merge_shots([], threshold=0.5, max_gap_s=3.0) == []
