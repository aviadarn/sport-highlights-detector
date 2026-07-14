import os
from nba_highlights.config import Weights
from nba_highlights.scoring.weighted import WeightedScorer


def load_scorer(fusion: str, model_path: str, weights: Weights):
    if fusion == "learned" and model_path and os.path.exists(model_path):
        from nba_highlights.scoring.learned import LearnedScorer
        return LearnedScorer(model_path)
    return WeightedScorer(weights)
