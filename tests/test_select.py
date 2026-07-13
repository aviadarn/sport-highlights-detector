from nba_highlights.select import select_highlights
from nba_highlights.models import Highlight, SignalBreakdown, AudioSignals, ActionSignals, Evidence


def _hl(hid, start, score):
    return Highlight(id=hid, start_s=start, end_s=start + 1, score=score,
                     signals=SignalBreakdown(audio=AudioSignals(), action=ActionSignals()),
                     evidence=Evidence(), member_scenes=[hid])


def test_filters_and_reorders_and_reids():
    hls = [_hl(0, 30.0, 0.9), _hl(1, 5.0, 0.4), _hl(2, 10.0, 0.6)]
    out = select_highlights(hls, threshold=0.5)
    assert [h.start_s for h in out] == [10.0, 30.0]
    assert [h.id for h in out] == [0, 1]


def test_empty():
    assert select_highlights([], threshold=0.5) == []
