from nba_highlights.interfaces import Word, TranscriptSegment, Transcript, ActionResult


def _transcript():
    return Transcript(segments=[TranscriptSegment(
        start_s=0.0, end_s=4.0, text="what a dunk oh my",
        words=[Word(text="what", start_s=0.0, end_s=0.5),
               Word(text="a", start_s=0.5, end_s=0.8),
               Word(text="dunk", start_s=1.0, end_s=1.4),
               Word(text="oh", start_s=3.0, end_s=3.2),
               Word(text="my", start_s=3.2, end_s=3.5)])])


def test_words_in_uses_midpoint_window():
    t = _transcript()
    picked = [w.text for w in t.words_in(0.9, 2.0)]
    assert picked == ["dunk"]


def test_text_in_joins_words():
    t = _transcript()
    assert t.text_in(2.5, 4.0) == "oh my"


def test_duration_is_last_segment_end():
    assert _transcript().duration_s == 4.0
    assert Transcript(segments=[]).duration_s == 0.0


def test_action_result_defaults():
    r = ActionResult(score=0.5, label="dunking basketball")
    assert r.top_k == []
