from typing import Protocol
from pydantic import BaseModel


class Word(BaseModel):
    text: str
    start_s: float
    end_s: float


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str
    words: list[Word]


class Transcript(BaseModel):
    segments: list[TranscriptSegment]

    @property
    def duration_s(self) -> float:
        return self.segments[-1].end_s if self.segments else 0.0

    def words_in(self, start_s: float, end_s: float) -> list[Word]:
        out: list[Word] = []
        for seg in self.segments:
            for w in seg.words:
                mid = (w.start_s + w.end_s) / 2.0
                if start_s <= mid < end_s:
                    out.append(w)
        return out

    def text_in(self, start_s: float, end_s: float) -> str:
        return " ".join(w.text for w in self.words_in(start_s, end_s)).strip()


class ActionResult(BaseModel):
    score: float
    label: str
    top_k: list[tuple[str, float]] = []


class Transcriber(Protocol):
    def transcribe(self, audio_path: str) -> Transcript: ...


class ActionRecognizer(Protocol):
    def score_clip(self, video_path: str, start_s: float, end_s: float) -> ActionResult: ...
