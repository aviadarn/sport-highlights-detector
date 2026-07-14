from nba_highlights.judge.base import JudgeInput, JudgeVerdict


def test_judge_input_fields():
    c = JudgeInput(scene_id=3, start_s=10.0, end_s=13.0,
                   keyframe_paths=["a.jpg", "b.jpg"], transcript="what a dunk")
    assert c.scene_id == 3 and c.keyframe_paths == ["a.jpg", "b.jpg"]


def test_judge_verdict_defaults():
    v = JudgeVerdict(is_highlight=True, confidence=0.8)
    assert v.play_type == "" and v.rationale == ""
    v2 = JudgeVerdict(is_highlight=False, confidence=0.1, play_type="timeout", rationale="dead ball")
    assert v2.play_type == "timeout"
