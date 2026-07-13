from pydantic import BaseModel


class VideoInfo(BaseModel):
    source: str
    title: str = ""
    duration_s: float = 0.0


class Shot(BaseModel):
    scene_id: int
    start_s: float
    end_s: float


class AudioSignals(BaseModel):
    loudness: float = 0.0
    prosody: float = 0.0
    keywords: float = 0.0


class ActionSignals(BaseModel):
    score: float = 0.0
    label: str = ""


class ShotScore(BaseModel):
    scene_id: int
    start_s: float
    end_s: float
    audio: AudioSignals
    action: ActionSignals
    audio_composite: float = 0.0
    fused: float = 0.0
    matched_keywords: list[str] = []
    peak_loudness_s: float | None = None
    transcript: str = ""


class SignalBreakdown(BaseModel):
    audio: AudioSignals
    action: ActionSignals


class Evidence(BaseModel):
    peak_loudness_s: float | None = None
    keywords: list[str] = []
    transcript: str = ""


class Highlight(BaseModel):
    id: int
    start_s: float
    end_s: float
    score: float
    signals: SignalBreakdown
    evidence: Evidence
    member_scenes: list[int]


class Report(BaseModel):
    video: VideoInfo
    config: dict
    generated_at: str
    highlights: list[Highlight]
