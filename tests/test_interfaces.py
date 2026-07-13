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


def test_words_in_excludes_midpoint_equal_to_end():
    """Guard against regression of < to <= in upper boundary check.

    Ensures that words whose midpoint equals end_s are EXCLUDED from the
    half-open interval [start_s, end_s).
    """
    t = Transcript(segments=[TranscriptSegment(
        start_s=0.0, end_s=3.0, text="control boundary beyond",
        words=[
            # Control word: midpoint 1.2 (in [0.0, 2.0))
            Word(text="control", start_s=1.0, end_s=1.4),
            # Boundary word: midpoint exactly 2.0 (should be excluded)
            Word(text="boundary", start_s=1.6, end_s=2.4),
            # Just below boundary: midpoint 1.98 (in [0.0, 2.0))
            Word(text="just_below", start_s=1.96, end_s=2.0),
            # Beyond: midpoint 2.5 (not in [0.0, 2.0))
            Word(text="beyond", start_s=2.4, end_s=2.6),
        ])])

    result = t.words_in(0.0, 2.0)
    result_texts = [w.text for w in result]

    # Include control and just_below, exclude boundary and beyond
    assert result_texts == ["control", "just_below"]
    assert "boundary" not in result_texts
    assert "beyond" not in result_texts


def test_action_result_defaults():
    r = ActionResult(score=0.5, label="dunking basketball")
    assert r.top_k == []
