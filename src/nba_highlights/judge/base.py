from typing import Protocol
from pydantic import BaseModel


class JudgeInput(BaseModel):
    scene_id: int
    start_s: float
    end_s: float
    keyframe_paths: list[str]
    transcript: str


class JudgeVerdict(BaseModel):
    is_highlight: bool
    confidence: float
    play_type: str = ""
    rationale: str = ""


class HighlightJudge(Protocol):
    def judge(self, clip: JudgeInput) -> JudgeVerdict: ...
