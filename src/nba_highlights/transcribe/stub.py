from nba_highlights.interfaces import Transcript


class StubTranscriber:
    def __init__(self, transcript: Transcript | None = None) -> None:
        self._transcript = transcript or Transcript(segments=[])

    def transcribe(self, audio_path: str) -> Transcript:
        return self._transcript
