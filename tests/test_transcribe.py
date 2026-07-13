from nba_highlights.transcribe.stub import StubTranscriber
from nba_highlights.interfaces import Transcript, TranscriptSegment, Word


def test_stub_returns_empty_by_default():
    assert StubTranscriber().transcribe("a.wav").segments == []


def test_stub_returns_injected_transcript():
    t = Transcript(segments=[TranscriptSegment(
        start_s=0.0, end_s=1.0, text="dunk",
        words=[Word(text="dunk", start_s=0.0, end_s=0.5)])])
    assert StubTranscriber(t).transcribe("a.wav").text_in(0.0, 1.0) == "dunk"
