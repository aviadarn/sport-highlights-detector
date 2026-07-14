from nba_highlights.eval import GroundTruth, TruthSegment
from nba_highlights.judge.agreement import agreement


def _gt(spans):
    return GroundTruth(highlights=[TruthSegment(start_s=s, end_s=e) for s, e in spans])


def test_perfect_agreement():
    g = _gt([(0.0, 10.0), (100.0, 110.0)])
    r = agreement(g, g)
    assert r.tp == 2 and r.fp == 0 and r.fn == 0
    assert r.precision == 1.0 and r.recall == 1.0 and r.f1 == 1.0


def test_partial_agreement_counts():
    judged = _gt([(0.0, 10.0), (200.0, 210.0)])   # first matches, second spurious
    human = _gt([(1.0, 11.0), (500.0, 510.0)])    # first matched, second missed
    r = agreement(judged, human, iou_threshold=0.5)
    assert r.tp == 1 and r.fp == 1 and r.fn == 1
    assert abs(r.precision - 0.5) < 1e-9 and abs(r.recall - 0.5) < 1e-9
    assert r.n_judged == 2 and r.n_human == 2
    assert abs(r.kappa - (-0.2)) < 1e-9   # tp=fp=fn=1: po=1/3, pe=4/9, kappa=-1/5


def test_empty_is_zero_not_crash():
    r = agreement(_gt([]), _gt([]))
    assert r.tp == 0 and r.f1 == 0.0 and r.kappa == 0.0
