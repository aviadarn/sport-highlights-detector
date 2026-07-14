import argparse
import json

from nba_highlights.training.evaluate import cross_validate
from nba_highlights.training.train import train_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the learned fusion model")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("FEATURES", "TRUTH"),
                    required=True, help="features.jsonl truth.json for one game (repeatable)")
    ap.add_argument("--out", required=True, help="output model path (.joblib)")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    pairs = [tuple(p) for p in args.pair]
    metrics = train_model(pairs, args.out, args.iou)
    # Leave-one-game-out comparison of the learned model vs the weighted baseline
    # (needs >=2 games so at least one can be held out).
    if len(pairs) >= 2:
        metrics["cv"] = cross_validate(pairs, iou_threshold=args.iou)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
