from nba_highlights.eval import iou, score_report, GroundTruth, TruthSegment
from nba_highlights.models import (
    Report, VideoInfo, Highlight, SignalBreakdown, AudioSignals, ActionSignals, Evidence,
)


def _hl(hid, start, end):
    return Highlight(id=hid, start_s=start, end_s=end, score=0.9,
                     signals=SignalBreakdown(audio=AudioSignals(), action=ActionSignals()),
                     evidence=Evidence(), member_scenes=[hid])


def _report(spans):
    return Report(video=VideoInfo(source="u"), config={}, generated_at="t",
                  highlights=[_hl(i, s, e) for i, (s, e) in enumerate(spans)])


def test_iou_basic():
    assert iou(0.0, 10.0, 0.0, 10.0) == 1.0
    assert iou(0.0, 10.0, 5.0, 15.0) == 5.0 / 15.0
    assert iou(0.0, 5.0, 10.0, 15.0) == 0.0


def test_score_report_tp_fp_fn():
    report = _report([(0.0, 10.0), (100.0, 110.0)])   # 1 good, 1 spurious
    gt = GroundTruth(highlights=[TruthSegment(start_s=1.0, end_s=11.0),   # matches first
                                 TruthSegment(start_s=200.0, end_s=210.0)])  # missed
    r = score_report(report, gt, iou_threshold=0.5)
    assert r.tp == 1 and r.fp == 1 and r.fn == 1
    assert abs(r.precision - 0.5) < 1e-9
    assert abs(r.recall - 0.5) < 1e-9
    assert abs(r.f1 - 0.5) < 1e-9


def test_perfect_match():
    report = _report([(0.0, 10.0)])
    gt = GroundTruth(highlights=[TruthSegment(start_s=0.0, end_s=10.0)])
    r = score_report(report, gt)
    assert r.tp == 1 and r.fp == 0 and r.fn == 0 and r.f1 == 1.0
