from typing import Protocol
from nba_highlights.features import ShotFeatures


class HighlightScorer(Protocol):
    def score(self, feats: ShotFeatures) -> float: ...
