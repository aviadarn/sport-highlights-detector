from nba_highlights.audio.keywords import shot_keywords
from nba_highlights.interfaces import Transcript, TranscriptSegment, Word

LEX = ["dunk", "and one", "three"]


def _t(words):
    return Transcript(segments=[TranscriptSegment(
        start_s=words[0][1], end_s=words[-1][2], text=" ".join(w[0] for w in words),
        words=[Word(text=w[0], start_s=w[1], end_s=w[2]) for w in words])])


def test_matches_single_and_multiword():
    t = _t([("what", 0.0, 0.3), ("a", 0.3, 0.5), ("dunk", 0.6, 1.0),
            ("and", 1.1, 1.3), ("one", 1.3, 1.6)])
    score, matched = shot_keywords(t, 0.0, 2.0, LEX)
    assert set(matched) == {"dunk", "and one"}
    assert score == 2.0


def test_no_words_in_window():
    t = _t([("dunk", 0.0, 0.5)])
    assert shot_keywords(t, 10.0, 11.0, LEX) == (0.0, [])


def test_case_insensitive():
    t = _t([("DUNK", 0.0, 0.5)])
    score, matched = shot_keywords(t, 0.0, 1.0, LEX)
    assert matched == ["dunk"] and score == 1.0
