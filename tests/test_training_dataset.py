import json
from nba_highlights.features import ShotFeatures
from nba_highlights.eval import GroundTruth, TruthSegment
from nba_highlights.training.dataset import label_shots, load_features, build_table


def test_label_shots_by_iou():
    feats = [ShotFeatures(scene_id=0, start_s=0.0, end_s=10.0),   # overlaps truth
             ShotFeatures(scene_id=1, start_s=100.0, end_s=110.0)]  # no truth
    gt = GroundTruth(highlights=[TruthSegment(start_s=1.0, end_s=11.0)])
    labeled = label_shots(feats, gt, iou_threshold=0.5)
    assert [lab for _, lab in labeled] == [1, 0]


def test_load_features_and_build_table(tmp_path):
    fpath = tmp_path / "g1.jsonl"
    fpath.write_text("\n".join(
        ShotFeatures(scene_id=i, start_s=float(i * 10), end_s=float(i * 10 + 10),
                     action_score=0.9 if i == 0 else 0.0).model_dump_json()
        for i in range(2)))
    tpath = tmp_path / "g1.truth.json"
    tpath.write_text(json.dumps({"highlights": [{"start_s": 0.0, "end_s": 10.0}]}))

    assert len(load_features(str(fpath))) == 2
    X, y, groups = build_table([(str(fpath), str(tpath))])
    assert len(X) == 2 and len(X[0]) == 7
    assert y == [1, 0]
    assert groups == [0, 0]
