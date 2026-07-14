from nba_highlights.judge.base import JudgeInput, JudgeVerdict


class StubHighlightJudge:
    def __init__(self, verdicts: dict[int, bool] | None = None,
                 default: bool = False, confidence: float = 1.0) -> None:
        self._verdicts = verdicts or {}
        self._default = default
        self._confidence = confidence

    def judge(self, clip: JudgeInput) -> JudgeVerdict:
        is_highlight = self._verdicts.get(clip.scene_id, self._default)
        return JudgeVerdict(
            is_highlight=is_highlight, confidence=self._confidence,
            play_type="stub" if is_highlight else "", rationale="stub")
