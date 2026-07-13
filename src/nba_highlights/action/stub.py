from nba_highlights.interfaces import ActionResult


class StubActionRecognizer:
    def __init__(self, scores: dict[int, float] | None = None,
                 default: float = 0.0) -> None:
        self._scores = scores or {}
        self._default = default

    def score_clip(self, video_path: str, start_s: float,
                   end_s: float) -> ActionResult:
        score = self._scores.get(round(start_s), self._default)
        return ActionResult(score=float(score),
                            label="stub-basketball" if score > 0 else "")
