from nba_highlights.models import (
    Report, VideoInfo, Highlight, SignalBreakdown, AudioSignals, ActionSignals, Evidence,
)
from nba_highlights.judge.base import JudgeVerdict
from nba_highlights.judge.labeling import build_judge_inputs, verdicts_to_truth


def _hl(hid, start, end, transcript):
    return Highlight(id=hid, start_s=start, end_s=end, score=0.9,
                     signals=SignalBreakdown(audio=AudioSignals(), action=ActionSignals()),
                     evidence=Evidence(transcript=transcript), member_scenes=[hid])


def _report(spans):
    return Report(video=VideoInfo(source="u"), config={}, generated_at="t",
                  highlights=[_hl(i, s, e, tx) for i, (s, e, tx) in enumerate(spans)])


def test_build_judge_inputs_extracts_and_carries_transcript(tmp_path):
    calls = []

    def fake_extract(video, t, out, runner=None):
        calls.append((t, out))
        return out

    report = _report([(0.0, 6.0, "what a dunk"), (30.0, 36.0, "from three")])
    inputs = build_judge_inputs(report, "game.mp4", str(tmp_path),
                                frames_per_clip=3, extract_frame_fn=fake_extract)
    assert len(inputs) == 2
    assert inputs[0].scene_id == 0 and inputs[0].transcript == "what a dunk"
    assert len(inputs[0].keyframe_paths) == 3          # 3 frames per clip
    assert len(calls) == 6                             # 2 clips x 3 frames


def test_verdicts_to_truth_filters_by_confidence():
    report = _report([(0.0, 6.0, "a"), (30.0, 36.0, "b"), (50.0, 55.0, "c")])
    verdicts = [
        JudgeVerdict(is_highlight=True, confidence=0.9),   # keep
        JudgeVerdict(is_highlight=True, confidence=0.3),   # drop: low confidence
        JudgeVerdict(is_highlight=False, confidence=0.9),  # drop: not a highlight
    ]
    gt = verdicts_to_truth(report, verdicts, min_confidence=0.5)
    assert [(s.start_s, s.end_s) for s in gt.highlights] == [(0.0, 6.0)]
