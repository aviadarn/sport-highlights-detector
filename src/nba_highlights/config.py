import os
from pydantic import BaseModel
from nba_highlights.errors import ConfigError

DEFAULT_HYPE_LEXICON = [
    "dunk", "dunks", "slam", "three", "threes", "and one", "and-one",
    "buzzer", "oh my", "are you kidding", "posterized", "block", "blocked",
    "steal", "alley oop", "alley-oop", "what a", "unbelievable", "clutch",
    "dagger", "and the foul", "from downtown", "count it",
]

ACTION_BACKENDS = {"stub", "videomae"}
ASR_BACKENDS = {"stub", "faster-whisper"}


class Weights(BaseModel):
    wL: float = 0.4
    wP: float = 0.3
    wK: float = 0.3
    wA: float = 0.5
    wV: float = 0.5


class Settings(BaseModel):
    weights: Weights = Weights()
    threshold: float = 0.5
    audio_gate: float = 0.35
    max_gap_s: float = 3.0
    action_backend: str = "stub"
    asr_backend: str = "stub"
    asr_model: str = "small"
    full_action: bool = False
    hype_lexicon: list[str] = DEFAULT_HYPE_LEXICON

    def validate(self) -> None:
        if self.action_backend not in ACTION_BACKENDS:
            raise ConfigError(f"unknown action_backend: {self.action_backend!r}")
        if self.asr_backend not in ASR_BACKENDS:
            raise ConfigError(f"unknown asr_backend: {self.asr_backend!r}")
        w = self.weights
        if abs((w.wL + w.wP + w.wK) - 1.0) > 1e-6:
            raise ConfigError("audio weights wL+wP+wK must sum to 1.0")
        if abs((w.wA + w.wV) - 1.0) > 1e-6:
            raise ConfigError("fusion weights wA+wV must sum to 1.0")

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.action_backend = os.environ.get("ACTION_BACKEND", s.action_backend)
        s.asr_backend = os.environ.get("ASR_BACKEND", s.asr_backend)
        s.asr_model = os.environ.get("ASR_MODEL", s.asr_model)
        if "THRESHOLD" in os.environ:
            s.threshold = float(os.environ["THRESHOLD"])
        if "AUDIO_GATE" in os.environ:
            s.audio_gate = float(os.environ["AUDIO_GATE"])
        if "MAX_GAP_S" in os.environ:
            s.max_gap_s = float(os.environ["MAX_GAP_S"])
        s.validate()
        return s
