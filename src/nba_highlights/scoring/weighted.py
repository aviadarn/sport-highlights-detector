from nba_highlights.config import Weights
from nba_highlights.features import ShotFeatures
from nba_highlights.fuse import audio_composite, fuse


class WeightedScorer:
    def __init__(self, weights: Weights) -> None:
        self._w = weights

    def score(self, feats: ShotFeatures) -> float:
        comp = audio_composite(feats.loudness_n, feats.prosody_n,
                               feats.keywords_n, self._w)
        return fuse(comp, feats.action_score, self._w)
