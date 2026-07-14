from nba_highlights.features import ShotFeatures


class StubScorer:
    def __init__(self, scores: dict[int, float] | None = None,
                 default: float = 0.0) -> None:
        self._scores = scores or {}
        self._default = default

    def score(self, feats: ShotFeatures) -> float:
        return self._scores.get(feats.scene_id, self._default)
