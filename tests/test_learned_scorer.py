from nba_highlights.features import ShotFeatures
from nba_highlights.config import Weights
from nba_highlights.scoring.learned import LearnedScorer
from nba_highlights.scoring.factory import load_scorer
from nba_highlights.scoring.weighted import WeightedScorer


class _FakeModel:
    def predict_proba(self, X):
        assert len(X) == 1 and len(X[0]) == 7  # one row, seven features
        return [[0.3, 0.7]]


def test_learned_scorer_returns_positive_class_proba():
    f = ShotFeatures(scene_id=0, start_s=0.0, end_s=1.0, action_score=0.9)
    assert LearnedScorer(model=_FakeModel()).score(f) == 0.7


def test_load_scorer_falls_back_to_weighted_when_model_missing(tmp_path):
    s = load_scorer("learned", str(tmp_path / "nope.joblib"), Weights())
    assert isinstance(s, WeightedScorer)


def test_load_scorer_weighted_when_fusion_weighted():
    assert isinstance(load_scorer("weighted", "", Weights()), WeightedScorer)


def test_load_scorer_learned_when_model_exists(tmp_path):
    p = tmp_path / "m.joblib"
    p.write_bytes(b"x")  # existence is enough; loading is lazy
    s = load_scorer("learned", str(p), Weights())
    assert isinstance(s, LearnedScorer)
