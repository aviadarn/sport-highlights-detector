from nba_highlights.features import ShotFeatures, to_vector


class LearnedScorer:
    def __init__(self, model_path: str | None = None, model=None) -> None:
        self._model_path = model_path
        self._model = model

    def _load(self):
        if self._model is None:
            import joblib
            self._model = joblib.load(self._model_path)
        return self._model

    def score(self, feats: ShotFeatures) -> float:
        model = self._load()
        return float(model.predict_proba([to_vector(feats)])[0][1])
