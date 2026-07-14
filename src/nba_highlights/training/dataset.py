from nba_highlights.eval import GroundTruth, iou
from nba_highlights.features import ShotFeatures, to_vector


def label_shots(features: list[ShotFeatures], gt: GroundTruth,
                iou_threshold: float = 0.5) -> list[tuple[ShotFeatures, int]]:
    out: list[tuple[ShotFeatures, int]] = []
    for f in features:
        positive = any(
            iou(f.start_s, f.end_s, t.start_s, t.end_s) >= iou_threshold
            for t in gt.highlights)
        out.append((f, 1 if positive else 0))
    return out


def load_features(path: str) -> list[ShotFeatures]:
    with open(path) as fh:
        return [ShotFeatures.model_validate_json(line)
                for line in fh if line.strip()]


def build_table(pairs: list[tuple[str, str]], iou_threshold: float = 0.5):
    X: list[list[float]] = []
    y: list[int] = []
    groups: list[int] = []
    for group_index, (feat_path, truth_path) in enumerate(pairs):
        feats = load_features(feat_path)
        with open(truth_path) as fh:
            gt = GroundTruth.model_validate_json(fh.read())
        for f, label in label_shots(feats, gt, iou_threshold):
            X.append(to_vector(f))
            y.append(label)
            groups.append(group_index)
    return X, y, groups
