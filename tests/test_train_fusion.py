import json
import pytest

pytest.importorskip("sklearn")

from nba_highlights.features import ShotFeatures
from nba_highlights.training.train import train_model


def _game(tmp_path, name, hot_scene):
    fp = tmp_path / f"{name}.jsonl"
    fp.write_text("\n".join(
        ShotFeatures(scene_id=i, start_s=float(i * 10), end_s=float(i * 10 + 10),
                     loudness_n=1.0 if i == hot_scene else 0.0,
                     action_score=1.0 if i == hot_scene else 0.0).model_dump_json()
        for i in range(4)))
    tp = tmp_path / f"{name}.truth.json"
    tp.write_text(json.dumps(
        {"highlights": [{"start_s": hot_scene * 10.0, "end_s": hot_scene * 10 + 10.0}]}))
    return (str(fp), str(tp))


def test_train_model_fits_and_persists(tmp_path):
    pairs = [_game(tmp_path, "g1", 1), _game(tmp_path, "g2", 2)]
    out = tmp_path / "model.joblib"
    metrics = train_model(pairs, str(out))
    assert out.exists()
    assert metrics["n"] == 8 and metrics["positives"] == 2
    assert metrics["cv_f1"] is not None

    # The persisted model loads and scores the separable signal correctly.
    import joblib
    from nba_highlights.features import to_vector
    model = joblib.load(str(out))
    hot = to_vector(ShotFeatures(scene_id=0, start_s=0, end_s=10,
                                 loudness_n=1.0, action_score=1.0))
    cold = to_vector(ShotFeatures(scene_id=0, start_s=0, end_s=10))
    assert model.predict_proba([hot])[0][1] > model.predict_proba([cold])[0][1]
