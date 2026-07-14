from nba_highlights.config import Weights
from nba_highlights.eval import GroundTruth
from nba_highlights.features import to_vector
from nba_highlights.scoring.weighted import WeightedScorer
from nba_highlights.training.dataset import label_shots, load_features


def _prf(y: list[int], pred: list[int]) -> dict:
    tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def _load_game(feat_path: str, truth_path: str, iou_threshold: float):
    feats = load_features(feat_path)
    with open(truth_path) as fh:
        gt = GroundTruth.model_validate_json(fh.read())
    y = [label for _, label in label_shots(feats, gt, iou_threshold)]
    return feats, y


def cross_validate(pairs: list[tuple[str, str]], iou_threshold: float = 0.5,
                   threshold: float = 0.5, weights: Weights | None = None) -> dict:
    """Leave-one-game-out shot-level comparison of the weighted baseline vs a
    learned model. For each held-out game, score every shot with
    ``WeightedScorer >= threshold`` and with a ``LogisticRegression`` trained on
    the other games, and report precision/recall/F1 against the shot labels.
    Returns per-game metrics plus the mean across games (learned averaged only
    over games where a two-class training split was available)."""
    from sklearn.linear_model import LogisticRegression

    weights = weights or Weights()
    wscorer = WeightedScorer(weights)
    games = [_load_game(f, t, iou_threshold) for f, t in pairs]

    per_game: list[dict] = []
    for i, (feats_h, y_h) in enumerate(games):
        weighted_pred = [1 if wscorer.score(f) >= threshold else 0 for f in feats_h]
        weighted_m = _prf(y_h, weighted_pred)

        x_train: list[list[float]] = []
        y_train: list[int] = []
        for j, (feats_o, y_o) in enumerate(games):
            if j == i:
                continue
            x_train.extend(to_vector(f) for f in feats_o)
            y_train.extend(y_o)

        learned_m = None
        if len(set(y_train)) == 2:  # need both classes to fit
            clf = LogisticRegression(max_iter=1000, C=1.0,
                                     class_weight="balanced").fit(x_train, y_train)
            learned_pred = list(clf.predict([to_vector(f) for f in feats_h]))
            learned_m = _prf(y_h, learned_pred)

        per_game.append({"game": i, "n": len(y_h), "positives": int(sum(y_h)),
                         "weighted": weighted_m, "learned": learned_m})

    def _mean(model: str, key: str):
        vals = [g[model][key] for g in per_game if g[model] is not None]
        return sum(vals) / len(vals) if vals else None

    mean = {model: {key: _mean(model, key)
                    for key in ("precision", "recall", "f1")}
            for model in ("weighted", "learned")}
    return {"games": per_game, "mean": mean, "n_games": len(games)}
