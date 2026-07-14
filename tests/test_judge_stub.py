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


def test_stub_branches_and_params():
    """Test non-highlight branch, default=True, and non-default confidence."""
    # Non-highlight: play_type should be empty, rationale still "stub"
    v_non_hl = StubHighlightJudge(verdicts={1: False}).judge(_clip(1))
    assert v_non_hl.is_highlight is False
    assert v_non_hl.play_type == ""
    assert v_non_hl.rationale == "stub"

    # default=True: unlisted scene_id should return True
    j_default_true = StubHighlightJudge(default=True)
    v_unlisted = j_default_true.judge(_clip(999))
    assert v_unlisted.is_highlight is True

    # Custom confidence: should use specified value, not default 1.0
    j_custom_conf = StubHighlightJudge(verdicts={1: True}, confidence=0.5)
    v_conf = j_custom_conf.judge(_clip(1))
    assert v_conf.confidence == 0.5
