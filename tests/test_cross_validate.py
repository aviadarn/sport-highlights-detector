import json

import pytest

pytest.importorskip("sklearn")

from nba_highlights.features import ShotFeatures
from nba_highlights.training.evaluate import cross_validate


def _game(tmp_path, name, hot_scene):
    """One game: 4 shots, `hot_scene` is loud+action and is the only truth segment."""
    fp = tmp_path / f"{name}.jsonl"
    fp.write_text("\n".join(
        ShotFeatures(scene_id=i, start_s=float(i * 10), end_s=float(i * 10 + 10),
                     loudness_n=1.0 if i == hot_scene else 0.0,
                     prosody_n=1.0 if i == hot_scene else 0.0,
                     keywords_n=1.0 if i == hot_scene else 0.0,
                     keyword_count=2.0 if i == hot_scene else 0.0,
                     action_score=1.0 if i == hot_scene else 0.0).model_dump_json()
        for i in range(4)))
    tp = tmp_path / f"{name}.truth.json"
    tp.write_text(json.dumps(
        {"highlights": [{"start_s": hot_scene * 10.0, "end_s": hot_scene * 10 + 10.0}]}))
    return (str(fp), str(tp))


def test_cross_validate_structure_and_means(tmp_path):
    pairs = [_game(tmp_path, "g1", 1), _game(tmp_path, "g2", 2), _game(tmp_path, "g3", 0)]
    result = cross_validate(pairs)

    assert result["n_games"] == 3
    assert len(result["games"]) == 3
    # every game reports both weighted and learned shot-level metrics
    for g in result["games"]:
        for model in ("weighted", "learned"):
            assert set(g[model]) == {"precision", "recall", "f1", "tp", "fp", "fn"}
    # means present for both models over the three keys
    for model in ("weighted", "learned"):
        assert set(result["mean"][model]) == {"precision", "recall", "f1"}
    # separable signal: learned achieves perfect held-out F1 on this toy data
    assert result["mean"]["learned"]["f1"] == 1.0


def test_cross_validate_learned_none_when_single_class(tmp_path):
    # Both games have NO positives -> training split is single-class -> learned=None.
    def _empty(name):
        fp = tmp_path / f"{name}.jsonl"
        fp.write_text("\n".join(
            ShotFeatures(scene_id=i, start_s=float(i), end_s=float(i + 1)).model_dump_json()
            for i in range(3)))
        tp = tmp_path / f"{name}.truth.json"
        tp.write_text(json.dumps({"highlights": []}))
        return (str(fp), str(tp))

    result = cross_validate([_empty("a"), _empty("b")])
    assert all(g["learned"] is None for g in result["games"])
    assert result["mean"]["learned"]["f1"] is None
