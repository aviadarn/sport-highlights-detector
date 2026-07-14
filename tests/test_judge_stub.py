from nba_highlights.judge.stub import StubHighlightJudge
from nba_highlights.judge.base import JudgeInput


def _clip(sid):
    return JudgeInput(scene_id=sid, start_s=0.0, end_s=1.0, keyframe_paths=[], transcript="")


def test_stub_keyed_by_scene_id():
    j = StubHighlightJudge(verdicts={1: True, 2: False}, default=False)
    assert j.judge(_clip(1)).is_highlight is True
    assert j.judge(_clip(2)).is_highlight is False
    assert j.judge(_clip(9)).is_highlight is False  # default


def test_stub_verdict_shape():
    v = StubHighlightJudge(verdicts={1: True}).judge(_clip(1))
    assert v.confidence == 1.0 and v.play_type == "stub"
