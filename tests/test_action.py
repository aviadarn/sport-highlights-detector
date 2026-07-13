from nba_highlights.action.basketball import BASKETBALL_CLASSES, score_from_probs
from nba_highlights.action.stub import StubActionRecognizer


def test_score_from_probs_picks_basketball_class():
    probs = {"dunking basketball": 0.7, "shooting basketball": 0.2, "typing": 0.1}
    r = score_from_probs(probs)
    assert r.score == 0.7
    assert r.label == "dunking basketball"
    assert r.top_k[0] == ("dunking basketball", 0.7)


def test_score_from_probs_zero_when_no_basketball():
    probs = {"typing": 0.9, "reading": 0.1}
    r = score_from_probs(probs)
    assert r.score == 0.0
    assert r.label == ""


def test_basketball_classes_nonempty():
    assert "dunking basketball" in BASKETBALL_CLASSES


def test_stub_action_keyed_by_start_second():
    rec = StubActionRecognizer(scores={12: 0.8}, default=0.1)
    hot = rec.score_clip("v.mp4", 12.2, 14.0)
    cold = rec.score_clip("v.mp4", 30.0, 31.0)
    assert hot.score == 0.8 and hot.label == "stub-basketball"
    assert cold.score == 0.1 and cold.label == "stub-basketball"


def test_stub_action_zero_label_empty():
    rec = StubActionRecognizer(default=0.0)
    assert rec.score_clip("v.mp4", 1.0, 2.0).label == ""
