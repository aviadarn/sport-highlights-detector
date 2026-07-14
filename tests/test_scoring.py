from nba_highlights.features import ShotFeatures
from nba_highlights.config import Weights
from nba_highlights.fuse import audio_composite, fuse
from nba_highlights.scoring.weighted import WeightedScorer
from nba_highlights.scoring.stub import StubScorer


def test_weighted_scorer_matches_m1_fusion():
    w = Weights()
    f = ShotFeatures(scene_id=0, start_s=0.0, end_s=2.0,
                     loudness_n=0.9, prosody_n=0.8, keywords_n=0.6, action_score=0.7)
    expected = fuse(audio_composite(0.9, 0.8, 0.6, w), 0.7, w)
    assert WeightedScorer(w).score(f) == expected


def test_stub_scorer_keyed_by_scene_id():
    s = StubScorer(scores={5: 0.9}, default=0.1)
    hot = ShotFeatures(scene_id=5, start_s=0.0, end_s=1.0)
    cold = ShotFeatures(scene_id=6, start_s=0.0, end_s=1.0)
    assert s.score(hot) == 0.9
    assert s.score(cold) == 0.1
